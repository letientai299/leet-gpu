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

void add_rate(
  nvbench::state& state,
  std::string tag,
  const char* name,
  const char* description,
  std::size_t work,
  double seconds
) {
  if (!std::isfinite(seconds) || seconds <= 0.0) {
    return;
  }
  auto& summary = state.add_summary(std::move(tag));
  summary.set_string("name", name);
  summary.set_string("description", description);
  summary.set_float64("value", static_cast<double>(work) / seconds / 1.0e9);
}

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

  nvbench::cuda_stream const stream;
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

// Non-Tensor Core FP32 FMA results per clock per SM.
// https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#arithmetic-instructions-throughput-native-arithmetic-instructions
int fp32_lanes_per_sm(int sm_version) {
  if (sm_version < 700) {
    return 0;
  }
  return sm_version <= 800 ? 64 : 128;
}

void add_compute_peak(nvbench::state& state, std::size_t flops, double seconds, double clock_hz) {
  const auto& device = state.get_device();
  if (!device.has_value() || !std::isfinite(seconds) || seconds <= 0.0) {
    return;
  }
  const int lanes = fp32_lanes_per_sm(device->get_sm_version());
  if (lanes == 0) {
    return;
  }
  // NVBench skips clock sampling under --run-once or --profile.
  if (!std::isfinite(clock_hz) || clock_hz <= 0.0) {
    return;
  }
  // An FMA counts as two FLOPs.
  const double peak = 2.0 * lanes * device->get_number_of_sms() * clock_hz;
  auto& peak_summary = state.add_summary("lg/fp32/peak");
  peak_summary.set_string("name", "FP32 Peak");
  peak_summary.set_string("description", "FP32 FMA FLOP/s at the measured clock");
  peak_summary.set_float64("value", peak);
  auto& summary = state.add_summary("lg/cold/fp32/utilization");
  summary.set_string("name", "FP32Util");
  summary.set_string("hint", "percentage");
  summary.set_string("description", "Cold FLOP/s over FP32 FMA peak at measured clock");
  summary.set_float64("value", static_cast<double>(flops) / seconds / peak);
}

// No-reuse bytes = DRAM bytes + bytes served on chip (L1, L2, shared memory).
// DRAM moves at least the minimum bytes and at most peak × time, which bounds
// the on-chip share.
void add_cache_bounds(nvbench::state& state, const Workload& workload, double seconds) {
  const auto& device = state.get_device();
  if (!workload.no_reuse || !device.has_value() || !std::isfinite(seconds) || seconds <= 0.0) {
    return;
  }
  const auto minimum =
    static_cast<double>(workload.minimum.read_bytes + workload.minimum.write_bytes);
  const auto no_reuse =
    static_cast<double>(workload.no_reuse->read_bytes + workload.no_reuse->write_bytes);
  const auto dram_limit = static_cast<double>(device->get_global_memory_bus_bandwidth()) * seconds;
  const auto add = [&](const char* tag, const char* name, double value) {
    auto& summary = state.add_summary(tag);
    summary.set_string("name", name);
    summary.set_string("hint", "percentage");
    summary.set_float64("value", std::clamp(value, 0.0, 1.0));
  };
  add("lg/cache/served_min", "Cache Served Min", 1.0 - dram_limit / no_reuse);
  add("lg/cache/served_max", "Cache Served Max", 1.0 - minimum / no_reuse);
}

void add_dram_peak(nvbench::state& state) {
  const auto& device = state.get_device();
  if (!device.has_value()) {
    return;
  }
  auto& summary = state.add_summary("lg/dram/peak");
  summary.set_string("name", "DRAM Peak");
  summary.set_string("hint", "byte_rate");
  summary.set_float64("value", static_cast<double>(device->get_global_memory_bus_bandwidth()));
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
  add_float(state, tag + "/intensity", "FLOP/B", flops / static_cast<double>(bytes));
}

} // namespace

bool describe(nvbench::state& state, const Workload& workload) {
  const auto minimum = total(workload.minimum);
  const auto no_reuse = workload.no_reuse ? total(*workload.no_reuse) : std::nullopt;
  if (!minimum || (workload.no_reuse && !no_reuse) || workload.flops == 0 ||
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
  add_summary(
    state, "lg/work/flops", "FLOPs", static_cast<nvbench::int64_t>(workload.flops), "flops"
  );
  add_summary(
    state, "lg/work/buffers", "Buffers", static_cast<nvbench::int64_t>(workload.buffer_bytes),
    "bytes"
  );
  add_traffic(state, "lg/traffic/minimum", "Minimum Bytes", *minimum, flops);
  if (no_reuse) {
    add_traffic(state, "lg/traffic/no_reuse", "No-Reuse Bytes", *no_reuse, flops);
  }
  // NVBench derives GlobalMem BW and BWUtil from these.
  state.add_global_memory_reads(workload.minimum.read_bytes);
  state.add_global_memory_writes(workload.minimum.write_bytes);
  return true;
}

void finish_summaries(nvbench::state& state, const Workload& workload) {
  double cold_seconds = 0.0;
  double batch_seconds = 0.0;
  ColdTiming timing;
  for (auto& summary : state.get_summaries()) {
    const auto& tag = summary.get_tag();
    if (tag == "nv/cold/time/gpu/mean") {
      cold_seconds = summary.get_float64("value");
    } else if (tag == "nv/batch/time/gpu/mean") {
      batch_seconds = summary.get_float64("value");
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

  add_rate(
    state, "lg/cold/gflops", "Cold GFLOPs/s",
    "Billions of floating-point operations per cold GPU second", workload.flops, cold_seconds
  );
  add_rate(
    state, "lg/batch/gflops", "Batch GFLOPs/s",
    "Billions of floating-point operations per batch GPU second", workload.flops, batch_seconds
  );
  add_compute_peak(state, workload.flops, cold_seconds, timing.clock_hz);
  add_dram_peak(state);
  add_cache_bounds(state, workload, cold_seconds);
  add_status(state, timing);
  if (!std::isfinite(cold_seconds) || cold_seconds <= 0 || timing.samples == 0 ||
      (!state.get_run_once() && timing.samples < state.get_min_samples())) {
    state.skip("insufficient valid timing samples");
  } else if (clock_unhealthy(state, timing.clock_hz)) {
    state.skip("GPU clock under half of max");
  }
}

void add_achieved_occupancy(
  nvbench::state& state, const std::function<void(cudaStream_t)>& launch
) {
  const auto& device = state.get_device();
  if (state.is_skipped() || !device.has_value()) {
    return;
  }
  const auto value =
    achieved_occupancy(device->get_id(), state.get_cuda_stream().get_stream(), launch);
  if (!value) {
    return;
  }
  auto& summary = state.add_summary("lg/occupancy/achieved");
  summary.set_string("name", "Achieved Occupancy");
  summary.set_string("hint", "percentage");
  summary.set_string("description", "Active warps per active SM cycle over the SM maximum");
  summary.set_float64("value", *value);
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
