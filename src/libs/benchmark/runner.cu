#include "benchmark/format.hpp"
#include "benchmark/profiler.hpp"
#include "benchmark/runner.hpp"
#include "validation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fmt/format.h>
#include <limits>
#include <map>
#include <nvbench/cuda_timer.cuh>
#include <nvbench/json_printer.cuh>
#include <nvbench/main.cuh>
#include <nvbench/option_parser.cuh>
#include <sstream>
#include <string>
#include <vector>

namespace lg::benchmark {
namespace {

inline constexpr nvbench::float32_t kThrottleThreshold = 0.75F;
inline constexpr nvbench::float64_t kTimeoutSeconds = 90.0;
inline constexpr nvbench::float64_t kWarmupSeconds = 1.0;

struct ColdTiming {
  double walltime = 0.0;
  double noise = std::numeric_limits<double>::infinity();
  double iqr = std::numeric_limits<double>::infinity();
  double iqr_seconds = std::numeric_limits<double>::infinity();
  double clock_hz = 0.0;
  double timer_step = 0.0;
  nvbench::int64_t samples = 0;
};

double infer_timer_step(std::vector<double> samples) {
  // Duration differences reveal quantization, not event overhead.
  std::ranges::sort(samples);
  samples.erase(std::unique(samples.begin(), samples.end()), samples.end());
  double step = std::numeric_limits<double>::infinity();
  for (std::size_t index = 1; index < samples.size(); ++index) {
    step = std::min(step, samples[index] - samples[index - 1]);
  }
  // Require repeated steps before relaxing the spread limit.
  int matches = 0;
  for (std::size_t index = 1; index < samples.size(); ++index) {
    if (std::abs(samples[index] - samples[index - 1] - step) <= step * 0.01) {
      ++matches;
    }
  }
  return std::isfinite(step) && matches >= 3 ? step : 0.0;
}

double event_timer_step() {
  int device = 0;
  NVBENCH_CUDA_CALL(cudaGetDevice(&device));
  static thread_local std::map<int, double> steps;
  if (const auto found = steps.find(device); found != steps.end()) {
    return found->second;
  }

  const nvbench::cuda_stream stream;
  nvbench::cuda_timer timer;
  std::vector<double> samples;
  samples.reserve(256);
  for (int sample = 0; sample < 256; ++sample) {
    timer.start(stream);
    // Vary intervals to expose timestamp quantization.
    for (int event = 0; event <= sample % 16; ++event) {
      timer.stop(stream);
    }
    const double seconds = timer.get_duration();
    if (std::isfinite(seconds) && seconds > 0.0) {
      samples.push_back(seconds);
    }
  }
  const double step = infer_timer_step(std::move(samples));
  steps.emplace(device, step);
  return step;
}

// deploy and bench export the clock they locked (gpu.sh).
double locked_clock_hz() {
  const char* value = std::getenv("LG_LOCKED_CLOCK_MHZ");
  if (value == nullptr) {
    return 0.0;
  }
  char* end = nullptr;
  const double mhz = std::strtod(value, &end);
  return end != value && std::isfinite(mhz) && mhz > 0.0 ? mhz * 1.0e6 : 0.0;
}

// A GPU that cannot hold half its max clock is faulty, not busy.
bool clock_unhealthy(nvbench::state& state, double clock_hz) {
  const auto& device = state.get_device();
  return device.has_value() && clock_hz > 0.0 &&
         clock_hz * 2.0 < static_cast<double>(device->get_sm_default_clock_rate());
}

// Clock problems come first: they invalidate every other verdict. Rare slow
// runs inflate the per-sample stdev without moving the result, so stability is
// judged by IQR / median and the mean's 95% confidence interval. An IQR of one
// timer step is the finest spread events can report, but only counts when that
// step is small next to the kernel time.
std::string cold_status(nvbench::state& state, const ColdTiming& timing) {
  constexpr double kDefaultMaxSpread = 0.005;
  constexpr double kMaxStepSpread = 0.05;
  // NVML clock samples wobble by a few MHz under a lock.
  constexpr double kClockTolerance = 0.99;
  const auto params = state.get_criterion_params();
  const double max_spread =
    params.has_value("max-noise") ? params.get_float64("max-noise") : kDefaultMaxSpread;
  const double locked_hz = locked_clock_hz();
  if (timing.samples == 0) {
    return "NO SAMPLES";
  }
  if (state.get_run_once()) {
    return "single run, spread n/a";
  }
  if (timing.samples < state.get_min_samples()) {
    return "INSUFFICIENT SAMPLES";
  }
  if (clock_unhealthy(state, timing.clock_hz)) {
    return fmt::format("UNHEALTHY · clock {:.3f} GHz, under half of max", timing.clock_hz / 1e9);
  }
  if (locked_hz > 0.0 && timing.clock_hz > 0.0 && timing.clock_hz < locked_hz * kClockTolerance) {
    return fmt::format(
      "THROTTLED · {:.3f}/{:.3f} GHz locked", timing.clock_hz / 1e9, locked_hz / 1e9
    );
  }
  if (state.get_timeout() > 0 && timing.walltime >= state.get_timeout()) {
    return fmt::format("TIMED OUT {:.0f}s", timing.walltime);
  }
  if (!std::isfinite(timing.noise) || timing.noise < 0 || !std::isfinite(timing.iqr) ||
      timing.iqr < 0) {
    return "spread unavailable";
  }
  const double interval = 1.96 * timing.noise / std::sqrt(static_cast<double>(timing.samples));
  const bool stable = timing.iqr <= max_spread ||
                      (timing.timer_step > 0.0 && timing.iqr_seconds <= timing.timer_step * 1.01 &&
                       timing.iqr <= kMaxStepSpread);
  return fmt::format(
    "{} · IQR {:.2f}% · mean ±{:.2f}% (95% CI)", stable ? "stable" : "UNSTABLE", timing.iqr * 100.0,
    interval * 100.0
  );
}

void add_status(nvbench::state& state, const ColdTiming& timing) {
  auto& summary = state.add_summary("lg/cold/status");
  summary.set_string("name", "Status");
  summary.set_string("description", "Whether the cold median is stable and clock-valid");
  summary.set_string("value", cold_status(state, timing));
}

} // namespace

nvbench::benchmark_base& configure(nvbench::benchmark_base& entry) {
  return entry.set_min_samples(20)
    .set_cold_warmup_runs(5)
    .set_cold_max_warmup_walltime(kWarmupSeconds)
    .set_batch_target_time(1.0)
    .set_timeout(kTimeoutSeconds)
    .set_throttle_threshold(kThrottleThreshold)
    .set_throttle_recovery_delay(0.1F);
}

void add_summary(
  nvbench::state& state, std::string tag, std::string name, nvbench::int64_t value, std::string hint
) {
  auto& summary = state.add_summary(std::move(tag));
  summary.set_string("name", std::move(name));
  if (!hint.empty()) {
    summary.set_string("hint", std::move(hint));
  }
  summary.set_int64("value", value);
}

void add_resources(nvbench::state& state, const KernelResources& resources) {
  auto& block = state.add_summary("kernel/resources/block");
  block.set_string("name", "Block");
  block.set_string(
    "value", fmt::format("{}×{}×{}", resources.block.x, resources.block.y, resources.block.z)
  );
  add_summary(
    state, "kernel/resources/threads_per_block", "Threads/Block", resources.threads_per_block
  );
  add_summary(
    state, "kernel/resources/registers_per_thread", "Registers/Thread",
    resources.registers_per_thread
  );
  add_summary(
    state, "kernel/resources/shared_static", "Static Shared Memory",
    static_cast<nvbench::int64_t>(resources.static_shared_bytes), "bytes"
  );
  add_summary(
    state, "kernel/resources/shared_dynamic", "Dynamic Shared Memory",
    static_cast<nvbench::int64_t>(resources.dynamic_shared_bytes), "bytes"
  );
  add_summary(
    state, "kernel/resources/constant", "Constant Memory",
    static_cast<nvbench::int64_t>(resources.constant_bytes), "bytes"
  );
  add_summary(
    state, "kernel/resources/local_per_thread", "Local Memory/Thread",
    static_cast<nvbench::int64_t>(resources.local_bytes_per_thread), "bytes"
  );
  if (resources.max_warps_per_sm > 0) {
    add_summary(
      state, "kernel/resources/blocks_per_sm", "Active Blocks/SM", resources.active_blocks_per_sm
    );
    add_summary(
      state, "kernel/resources/warps_per_sm", "Active Warps/SM", resources.active_warps_per_sm
    );
    add_summary(
      state, "kernel/resources/max_warps_per_sm", "Max Warps/SM", resources.max_warps_per_sm
    );
    auto& occupancy = state.add_summary("kernel/resources/occupancy");
    occupancy.set_string("name", "Theoretical Occupancy");
    occupancy.set_string("hint", "percentage");
    occupancy.set_float64(
      "value", static_cast<double>(resources.active_warps_per_sm) / resources.max_warps_per_sm
    );
  }
}

bool add_resources(
  nvbench::state& state, const std::function<cudaError_t(KernelResources&)>& query
) {
  KernelResources resources;
  if (!check_status(query(resources))) {
    state.skip("CUDA kernel resource query failed");
    return false;
  }
  add_resources(state, resources);
  return true;
}

namespace {

std::optional<std::size_t> total(const MemoryTraffic& traffic) {
  const auto bytes = lg::checked_add(traffic.read_bytes, traffic.write_bytes);
  constexpr auto limit = static_cast<std::size_t>(std::numeric_limits<nvbench::int64_t>::max());
  return bytes && *bytes != 0 && *bytes <= limit ? bytes : std::nullopt;
}

void add_float(
  nvbench::state& state, std::string tag, const char* name, double value, const char* hint = ""
) {
  auto& summary = state.add_summary(std::move(tag));
  summary.set_string("name", name);
  if (*hint != '\0') {
    summary.set_string("hint", hint);
  }
  summary.set_float64("value", value);
}

void add_traffic(
  nvbench::state& state, const std::string& tag, const char* name, std::size_t bytes, double flops
) {
  add_summary(state, tag + "/bytes", name, static_cast<nvbench::int64_t>(bytes), "bytes");
  if (flops > 0) {
    add_float(state, tag + "/intensity", "FLOP/B", flops / static_cast<double>(bytes));
  }
}

} // namespace

bool describe(nvbench::state& state, const Workload& workload) {
  const auto minimum = total(workload.minimum);
  const auto no_reuse = workload.no_reuse ? total(*workload.no_reuse) : std::nullopt;
  if (!minimum || (workload.no_reuse && !no_reuse) ||
      workload.flops > static_cast<std::size_t>(std::numeric_limits<nvbench::int64_t>::max())) {
    state.skip("invalid workload size");
    return false;
  }
  if (!state.get_run_once()) {
    add_float(state, "lg/cold/timer_step", "Event Timer Step", event_timer_step(), "duration");
  }
  for (const auto& [name, value] : workload.shape) {
    add_summary(state, "lg/shape/" + name, name, value);
  }
  const auto flops = static_cast<double>(workload.flops);
  if (workload.flops > 0) {
    add_summary(
      state, "lg/work/flops", "Modeled FLOPs", static_cast<nvbench::int64_t>(workload.flops),
      "flops"
    );
  }
  add_summary(
    state, "lg/work/buffers", "Buffers", static_cast<nvbench::int64_t>(workload.buffer_bytes),
    "bytes"
  );
  add_traffic(state, "lg/traffic/minimum", "Minimum Bytes", *minimum, flops);
  if (no_reuse) {
    add_traffic(state, "lg/traffic/no_reuse", "No-Reuse Bytes", *no_reuse, flops);
  }
  return true;
}

void finish_summaries(nvbench::state& state) {
  double cold_seconds = 0.0;
  ColdTiming timing;
  for (auto& summary : state.get_summaries()) {
    const auto& tag = summary.get_tag();
    if (tag == "nv/cold/time/gpu/mean") {
      cold_seconds = summary.get_float64("value");
    } else if (tag == "nv/cold/walltime") {
      timing.walltime = summary.get_float64("value");
    } else if (tag == "nv/cold/time/gpu/stdev/relative") {
      timing.noise = summary.get_float64("value");
    } else if (tag == "nv/cold/sample_size") {
      timing.samples = summary.get_int64("value");
    } else if (tag == "nv/cold/sm_clock_rate/mean") {
      timing.clock_hz = summary.get_float64("value");
    } else if (tag == "nv/cold/time/gpu/iqr/relative") {
      timing.iqr = summary.get_float64("value");
    } else if (tag == "nv/cold/time/gpu/iqr/absolute") {
      timing.iqr_seconds = summary.get_float64("value");
    } else if (tag == "lg/cold/timer_step") {
      timing.timer_step = summary.get_float64("value");
    }
  }

  add_status(state, timing);
  if (!std::isfinite(cold_seconds) || cold_seconds <= 0 || timing.samples == 0 ||
      (!state.get_run_once() && timing.samples < state.get_min_samples())) {
    state.skip("insufficient valid timing samples");
  } else if (clock_unhealthy(state, timing.clock_hz)) {
    state.skip("GPU clock under half of max");
  }
}

namespace {

void add_compute_metrics(nvbench::state& state, const HardwareMetrics& metrics) {
  const auto add_compute = [&](std::string_view kind, double count, double rate, double peak) {
    if (count <= 0 || peak <= 0) {
      return;
    }
    const auto tag = std::string("lg/hardware/") + std::string(kind);
    const auto* name = kind == "fp32" ? "Measured FP32 FLOPs" : "Measured FP64 FLOPs";
    add_float(state, tag + "/flops", name, count, "flops");
    add_float(state, tag + "/rate", "Measured FLOP/s", rate);
    add_float(state, tag + "/peak", "Sustained Peak", peak);
    add_float(state, tag + "/utilization", "Measured Utilization", rate / peak, "percentage");
  };
  add_compute("fp32", metrics.fp32_flops, metrics.fp32_flops_per_second, metrics.fp32_peak);
  add_compute("fp64", metrics.fp64_flops, metrics.fp64_flops_per_second, metrics.fp64_peak);
  const auto add_activity = [&](std::string_view kind, double utilization) {
    if (utilization > 0) {
      add_float(
        state, std::string("lg/hardware/") + std::string(kind) + "/utilization",
        "Measured Utilization", utilization, "percentage"
      );
    }
  };
  add_activity("fp16", metrics.fp16_utilization);
  add_activity("tensor", metrics.tensor_utilization);
}

void add_integer_metrics(nvbench::state& state, const HardwareMetrics& metrics) {
  if (metrics.integer_instructions > 0) {
    add_float(
      state, "lg/hardware/integer/instructions", "Integer Thread Instructions",
      metrics.integer_instructions
    );
    add_float(
      state, "lg/hardware/integer/rate", "Integer Thread Instructions/s",
      metrics.integer_instructions_per_second
    );
    add_float(
      state, "lg/hardware/integer/utilization", "Integer ALU Utilization",
      metrics.integer_utilization, "percentage"
    );
  }
}

void add_atomic_metrics(nvbench::state& state, const HardwareMetrics& metrics) {
  if (metrics.global_atomics == 0 && metrics.global_reductions == 0 &&
      metrics.shared_atomics == 0 && metrics.atomic_sectors == 0 &&
      metrics.reduction_sectors == 0) {
    return;
  }
  add_float(
    state, "lg/hardware/atomics/global", "Global ATOM Warp Instructions", metrics.global_atomics
  );
  add_float(
    state, "lg/hardware/atomics/reductions", "Global RED Warp Instructions",
    metrics.global_reductions
  );
  add_float(
    state, "lg/hardware/atomics/shared", "Shared Atomic Warp Instructions", metrics.shared_atomics
  );
  const auto add_atomic =
    [&](std::string_view kind, double count, double rate, double utilization) {
      if (count <= 0) {
        return;
      }
      const auto tag = std::string("lg/hardware/") + std::string(kind);
      const auto* name = kind == "atomics" ? "L2 ATOM Sectors" : "L2 RED Sectors";
      add_float(state, tag + "/sectors", name, count);
      add_float(state, tag + "/rate", "L2 Atomic Sectors/s", rate);
      add_float(state, tag + "/utilization", "L2 Atomic Utilization", utilization, "percentage");
    };
  add_atomic(
    "atomics", metrics.atomic_sectors, metrics.atomic_sectors_per_second, metrics.atomic_utilization
  );
  add_atomic(
    "reductions", metrics.reduction_sectors, metrics.reduction_sectors_per_second,
    metrics.reduction_utilization
  );
  if (metrics.shared_atomics > 0) {
    add_float(
      state, "lg/hardware/shared_atomics/utilization", "Shared Atomic Instruction Utilization",
      metrics.shared_atomic_utilization, "percentage"
    );
  }
}

} // namespace

void add_hardware_metrics(nvbench::state& state, const std::function<void(cudaStream_t)>& launch) {
  const auto& device = state.get_device();
  if (state.is_skipped() || !device.has_value()) {
    return;
  }
  const auto metrics =
    profile_hardware(device->get_id(), state.get_cuda_stream().get_stream(), launch);
  if (!metrics) {
    auto& status = state.add_summary("lg/hardware/status");
    status.set_string("name", "Hardware Counters");
    status.set_string("value", "unavailable");
    return;
  }
  add_float(state, "lg/occupancy/achieved", "Achieved Occupancy", metrics->occupancy, "percentage");
  add_float(
    state, "lg/hardware/dram/read_bytes", "Measured DRAM Reads", metrics->dram_read_bytes, "bytes"
  );
  add_float(
    state, "lg/hardware/dram/write_bytes", "Measured DRAM Writes", metrics->dram_write_bytes,
    "bytes"
  );
  add_float(
    state, "lg/hardware/dram/rate", "Measured DRAM BW", metrics->dram_bytes_per_second, "byte_rate"
  );
  add_float(
    state, "lg/hardware/dram/utilization", "Measured DRAM Utilization", metrics->dram_utilization,
    "percentage"
  );
  add_float(state, "lg/hardware/dram/peak", "DRAM Sustained Peak", metrics->dram_peak, "byte_rate");
  add_compute_metrics(state, *metrics);
  add_integer_metrics(state, *metrics);
  add_atomic_metrics(state, *metrics);
}

inline int run_impl(int argc, char** argv) try {
  nvbench::detail::main_initialize(argc, argv);
  auto args = nvbench::detail::main_convert_args(argc, argv);
  nvbench::option_parser parser;
  parser.set_raw_args(args);
  parser.parse(std::move(args));
  nvbench::detail::main_print_preamble(parser);
  nvbench::detail::main_run_benchmarks(parser);
  nvbench::detail::main_print_epilogue(parser);
  nvbench::detail::main_print_results(parser);
  return 0;
}
NVBENCH_MAIN_CATCH_EXCEPTIONS

bool has_option(int argc, char** argv, std::initializer_list<std::string_view> options) {
  for (int index = 1; index < argc; ++index) {
    const std::string_view arg = argv[index];
    if (std::ranges::find(options, arg) != options.end()) {
      return true;
    }
  }
  return false;
}

std::vector<std::string> benchmark_args(int argc, char** argv) {
  auto raw_args = nvbench::detail::main_convert_args(argc, argv);
  std::vector<std::string> args;
  args.reserve(raw_args.size() + 2);
  args.push_back(raw_args.front());
  args.emplace_back("--devices");
  args.emplace_back("0");
  args.insert(args.end(), raw_args.begin() + 1, raw_args.end());
  return args;
}

std::vector<char*> arg_pointers(std::vector<std::string>& args) {
  std::vector<char*> pointers;
  pointers.reserve(args.size());
  for (auto& arg : args) {
    pointers.push_back(arg.data());
  }
  return pointers;
}

int run_captured(int argc, char** argv, std::string& output) try {
  nvbench::detail::main_initialize(argc, argv);
  int result = 0;
  {
    auto raw_args = nvbench::detail::main_convert_args(argc, argv);
    auto args = raw_args;
    args.emplace_back("--quiet");

    nvbench::option_parser parser;
    parser.set_raw_args(raw_args);
    parser.parse(std::move(args));
    nvbench::detail::main_run_benchmarks(parser);
    nvbench::detail::main_print_results(parser);
    std::size_t states = 0;
    for (const auto& bench : parser.get_benchmarks()) {
      for (const auto& state : bench->get_states()) {
        ++states;
        if (state.is_skipped()) {
          fmt::print(stderr, "Benchmark skipped: {}\n", state.get_skip_reason());
          result = 1;
        }
      }
    }
    if (states == 0) {
      fmt::print(stderr, "No benchmark states ran\n");
      result = 1;
    }

    std::ostringstream stream;
    nvbench::json_printer printer(stream);
    printer.log_raw_argv(raw_args);
    printer.log_argv(raw_args);
    printer.print_benchmark_results(parser.get_benchmarks());
    output = stream.str();
  }
  // Callers still own CUDA buffers and contexts.
  return result;
}
NVBENCH_MAIN_CATCH_EXCEPTIONS

int run_args(int argc, char** argv) try {
  if (has_option(
        argc, argv,
        {"--help", "-h", "--help-axis", "--help-axes", "--version", "--list", "-l",
         "--jsonlist-benches", "--jsonlist-devices"}
      )) {
    return run_impl(argc, argv);
  }

  auto args = benchmark_args(argc, argv);
  auto pointers = arg_pointers(args);
  const int bench_argc = static_cast<int>(pointers.size());
  char** bench_argv = pointers.data();
  std::string output;
  const int result = run_captured(bench_argc, bench_argv, output);
  if (!output.empty() && !has_option(argc, argv, {"--json", "--jsonbin", "--quiet", "-q"})) {
    print_result(output);
  }
  return result;
} catch (const std::exception& error) {
  fmt::print(stderr, "{}\n", error.what());
  return 1;
}

} // namespace lg::benchmark
