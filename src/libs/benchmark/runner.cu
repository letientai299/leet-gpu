#include "benchmark/format.hpp"
#include "benchmark/runner.hpp"

#include <algorithm>
#include <cmath>
#include <fmt/format.h>
#include <limits>
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

void add_rate(nvbench::state& state,
              std::string tag,
              const char* name,
              const char* description,
              std::size_t work,
              double seconds) {
  if (seconds <= 0.0) {
    return;
  }
  auto& summary = state.add_summary(std::move(tag));
  summary.set_string("name", name);
  summary.set_string("description", description);
  summary.set_float64("value", static_cast<double>(work) / seconds / 1.0e9);
}

void add_status(nvbench::state& state,
                std::string tag,
                double walltime,
                double noise,
                nvbench::int64_t samples) {
  const auto params = state.get_criterion_params();
  const double max_noise = params.has_value("max-noise") ? params.get_float64("max-noise") : 0.0;
  std::string status;
  if (walltime > state.get_timeout() * 1.001) {
    status = fmt::format("TIMED OUT {:.0f}s", walltime);
  } else if (!std::isfinite(noise)) {
    status = "converged, noise n/a";
  } else if (noise < max_noise) {
    status = "converged";
  } else {
    status = "UNSTABLE, noise plateaued";
  }
  auto& summary = state.add_summary(std::move(tag));
  summary.set_string("name", "Status");
  summary.set_string("description", "Whether cold timing met its noise target");
  summary.set_string("value", fmt::format("{} ({}x)", status, samples));
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

void add_summary(nvbench::state& state,
                 std::string tag,
                 std::string name,
                 nvbench::int64_t value,
                 std::string hint) {
  auto& summary = state.add_summary(std::move(tag));
  summary.set_string("name", std::move(name));
  if (!hint.empty()) {
    summary.set_string("hint", std::move(hint));
  }
  summary.set_int64("value", value);
}

void add_resources(nvbench::state& state, const KernelResources& resources) {
  add_summary(state, "kernel/resources/threads_per_block", "Threads/Block",
              resources.threads_per_block);
  add_summary(state, "kernel/resources/registers_per_thread", "Registers/Thread",
              resources.registers_per_thread);
  add_summary(state, "kernel/resources/shared_static", "Static Shared Memory",
              static_cast<nvbench::int64_t>(resources.static_shared_bytes), "bytes");
  add_summary(state, "kernel/resources/shared_dynamic", "Dynamic Shared Memory",
              static_cast<nvbench::int64_t>(resources.dynamic_shared_bytes), "bytes");
  add_summary(state, "kernel/resources/constant", "Constant Memory",
              static_cast<nvbench::int64_t>(resources.constant_bytes), "bytes");
  add_summary(state, "kernel/resources/local_per_thread", "Local Memory/Thread",
              static_cast<nvbench::int64_t>(resources.local_bytes_per_thread), "bytes");
}

void finish_summaries(nvbench::state& state,
                      std::string_view prefix,
                      std::size_t flops,
                      std::optional<std::size_t> model_bytes) {
  double cold_seconds = 0.0;
  double batch_seconds = 0.0;
  double walltime = 0.0;
  double noise = std::numeric_limits<double>::infinity();
  nvbench::int64_t samples = 0;
  for (auto& summary : state.get_summaries()) {
    const auto tag = summary.get_tag();
    if (tag == "nv/cold/time/gpu/mean") {
      cold_seconds = summary.get_float64("value");
    } else if (tag == "nv/batch/time/gpu/mean") {
      batch_seconds = summary.get_float64("value");
    } else if (tag == "nv/cold/walltime") {
      walltime = summary.get_float64("value");
    } else if (tag == "nv/cold/time/gpu/stdev/relative") {
      noise = summary.get_float64("value");
    } else if (tag == "nv/cold/sample_size") {
      samples = summary.get_int64("value");
    } else if (tag == "nv/cold/sm_clock_rate/mean" ||
               tag == "nv/cold/sm_clock_rate/scaling/percent") {
      summary.remove_value("hide");
    }
  }

  const std::string base(prefix);
  add_rate(state, base + "/cold/gflops", "Cold GFLOPs/s",
           "Billions of floating-point operations per cold GPU second", flops, cold_seconds);
  add_rate(state, base + "/batch/gflops", "Batch GFLOPs/s",
           "Billions of floating-point operations per batch GPU second", flops, batch_seconds);
  if (model_bytes) {
    add_rate(state, base + "/cold/model_gbytes", "Cold Model GB/s",
             "Modeled global-memory bytes per cold GPU second", *model_bytes, cold_seconds);
    add_rate(state, base + "/batch/model_gbytes", "Batch Model GB/s",
             "Modeled global-memory bytes per batch GPU second", *model_bytes, batch_seconds);
  }
  add_status(state, base + "/cold/status", walltime, noise, samples);
}

inline int run_impl(int argc, char** argv) try { NVBENCH_MAIN_BODY(argc, argv); }
NVBENCH_MAIN_CATCH_EXCEPTIONS

bool has_option(int argc, char** argv, std::initializer_list<std::string_view> options) {
  for (int index = 1; index < argc; ++index) {
    const std::string_view arg = argv[index];
    if (std::find(options.begin(), options.end(), arg) != options.end()) {
      return true;
    }
  }
  return false;
}

std::vector<std::string> benchmark_args(int argc, char** argv) {
  auto raw_args = nvbench::detail::main_convert_args(argc, argv);
  std::vector<std::string> args;
  args.reserve(raw_args.size() + 4);
  args.push_back(raw_args.front());
  args.emplace_back("--devices");
  args.emplace_back("0");
  args.emplace_back("--throttle-threshold");
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
  {
    auto raw_args = nvbench::detail::main_convert_args(argc, argv);
    auto args = raw_args;
    args.emplace_back("--quiet");

    nvbench::option_parser parser;
    parser.set_raw_args(raw_args);
    parser.parse(std::move(args));
    nvbench::detail::main_run_benchmarks(parser);

    std::ostringstream stream;
    nvbench::json_printer printer(stream);
    printer.log_raw_argv(raw_args);
    printer.log_argv(raw_args);
    printer.print_benchmark_results(parser.get_benchmarks());
    output = stream.str();
  }
  nvbench::detail::main_finalize();
  return 0;
}
NVBENCH_MAIN_CATCH_EXCEPTIONS

int run_json(int argc, char** argv) {
  auto args = nvbench::detail::main_convert_args(argc, argv);
  args.emplace_back("--quiet");
  auto pointers = arg_pointers(args);
  return run_impl(static_cast<int>(pointers.size()), pointers.data());
}

int run_args(int argc, char** argv) {
  if (has_option(argc, argv,
                 {"--help", "-h", "--help-axis", "--help-axes", "--version", "--list", "-l",
                  "--jsonlist-benches", "--jsonlist-devices"})) {
    return run_impl(argc, argv);
  }

  auto args = benchmark_args(argc, argv);
  auto pointers = arg_pointers(args);
  const int bench_argc = static_cast<int>(pointers.size());
  char** bench_argv = pointers.data();
  if (has_option(argc, argv, {"--json", "--jsonbin"})) {
    return run_json(bench_argc, bench_argv);
  }
  if (has_option(argc, argv, {"--quiet", "-q"})) {
    return run_impl(bench_argc, bench_argv);
  }

  std::string output;
  const int result = run_captured(bench_argc, bench_argv, output);
  if (result == 0) {
    print_result(output);
  }
  return result;
}

} // namespace lg::benchmark
