#include "checks.hpp"
#include "log.hpp"
#include "matmul/app.hpp"
#include "matmul/benchmark.hpp"
#include "matmul/correctness.hpp"

#include <cstdio>
#include <cuda/buffer>
#include <limits>
#include <map>
#include <memory>
#include <nvbench/benchmark.cuh>
#include <nvbench/benchmark_manager.cuh>
#include <nvbench/type_list.cuh>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lg::matmul {
namespace {

using DeviceMatrix = cuda::device_buffer<float>;

struct FixedRunner {
  Benchmark* harness;
  Kernel kernel;

  void operator()(nvbench::state& state, nvbench::type_list<>) const {
    harness->run_with_shape(state, kernel);
  }
};

struct ShapeRunner {
  Kernel kernel;

  void operator()(nvbench::state& state, nvbench::type_list<>) const {
    const auto shape = get_shape(state);
    if (shape.has_value()) {
      benchmark(state, *shape, kernel);
    }
  }
};

template <typename Runner> nvbench::benchmark_base& add_benchmark(Kernel kernel, Runner runner) {
  auto entry = std::make_unique<nvbench::benchmark<Runner>>(std::move(runner));
  return nvbench::benchmark_manager::get()
      .add(std::move(entry))
      .set_name(std::string(kernel.name))
      .set_min_samples(20)
      .set_cold_warmup_runs(5)
      .set_batch_target_time(1.0)
      .set_throttle_threshold(0.9F)
      .set_throttle_recovery_delay(0.1F);
}

void add_fixed(Benchmark& harness, Kernel kernel) {
  add_benchmark(kernel, FixedRunner{&harness, kernel});
}

void add_shaped(Kernel kernel) {
  add_benchmark(kernel, ShapeRunner{kernel})
      .add_int64_axis("Height", {kDefaultHeight})
      .add_int64_axis("Width", {kDefaultWidth})
      .add_int64_axis("K", {kDefaultK});
}

void print_app_usage(const char* app, const GemmShape& bench_shape) {
  std::printf("Usage: %s [--height N] [--width N] [--k N] [--bench [options]]\n", app);
  std::printf("Check defaults: --height %u --width %u --k %u\n", kDefaultHeight, kDefaultWidth,
              kDefaultK);
  std::printf("Bench defaults: --height %u --width %u --k %u\n", bench_shape.m(), bench_shape.n(),
              bench_shape.k());
}

void apply_bench_shape(AppArgs& args, const GemmShape& shape) {
  if (!args.has_height) {
    args.height = shape.m();
  }
  if (!args.has_width) {
    args.width = shape.n();
  }
  if (!args.has_k) {
    args.k = shape.k();
  }
}

void print_choice_usage(const char* app, std::initializer_list<KernelChoice> choices) {
  std::printf("Usage: %s [--kernel ", app);
  const char* separator = "";
  for (const auto& choice : choices) {
    std::printf("%s%.*s", separator, static_cast<int>(choice.option.size()), choice.option.data());
    separator = "|";
  }
  std::printf("] [--height N] [--width N] [--k N] [--bench [options]]\n");
  std::printf("Defaults: --height %u --width %u --k %u\n", kDefaultHeight, kDefaultWidth,
              kDefaultK);
}

const KernelChoice* find_choice(std::initializer_list<KernelChoice> choices,
                                std::string_view name) {
  for (const auto& choice : choices) {
    if (choice.option == name) {
      return &choice;
    }
  }
  return nullptr;
}

bool parse_choice_args(int argc,
                       char** argv,
                       std::initializer_list<KernelChoice> choices,
                       const KernelChoice*& selected,
                       std::vector<char*>& remaining) {
  remaining.push_back(argv[0]);
  for (int index = 1; index < argc; ++index) {
    const std::string_view arg = argv[index];
    if (arg == "--kernel") {
      ++index;
      if (index == argc) {
        return false;
      }
      selected = find_choice(choices, argv[index]);
      if (selected == nullptr) {
        return false;
      }
      continue;
    }
    constexpr std::string_view prefix = "--kernel=";
    if (arg.rfind(prefix, 0) == 0) {
      selected = find_choice(choices, arg.substr(prefix.size()));
      if (selected == nullptr) {
        return false;
      }
      continue;
    }
    remaining.push_back(argv[index]);
  }
  return true;
}

struct DeviceData {
  GemmShape shape;
  DeviceMatrix a;
  DeviceMatrix b;
  DeviceMatrix c;

  DeviceData(const Problem& problem, int device)
      : shape(problem.shape),
        a(default_stream(), cuda::device_default_memory_pool(cuda::devices[device]), problem.a),
        b(default_stream(), cuda::device_default_memory_pool(cuda::devices[device]), problem.b),
        c(default_stream(),
          cuda::device_default_memory_pool(cuda::devices[device]),
          problem.result) {
  }
};

void add_summary(nvbench::state& state, std::string tag, std::string name, nvbench::int64_t value) {
  auto& summary = state.add_summary(std::move(tag));
  summary.set_string("name", std::move(name));
  summary.set_int64("value", value);
}

void finish_summaries(nvbench::state& state, std::size_t flops) {
  double cold_seconds = 0.0;
  double batch_seconds = 0.0;
  for (auto& summary : state.get_summaries()) {
    if (summary.get_tag() == "nv/cold/time/gpu/mean") {
      cold_seconds = summary.get_float64("value");
    } else if (summary.get_tag() == "nv/batch/time/gpu/mean") {
      batch_seconds = summary.get_float64("value");
    } else if (summary.get_tag() == "nv/cold/sm_clock_rate/mean" ||
               summary.get_tag() == "nv/cold/sm_clock_rate/scaling/percent") {
      summary.remove_value("hide");
    }
  }

  if (cold_seconds > 0.0) {
    auto& summary = state.add_summary("matmul/cold/gflops");
    summary.set_string("name", "Cold GFLOPs/s");
    summary.set_string("description", "Billions of floating-point operations per cold GPU second");
    summary.set_float64("value", static_cast<double>(flops) / cold_seconds / 1.0e9);
  }
  if (batch_seconds > 0.0) {
    auto& summary = state.add_summary("matmul/batch/gflops");
    summary.set_string("name", "Batch GFLOPs/s");
    summary.set_string("description", "Billions of floating-point operations per batch GPU second");
    summary.set_float64("value", static_cast<double>(flops) / batch_seconds / 1.0e9);
  }
}

bool get_counts(const GemmShape& shape,
                std::size_t& outputs,
                std::size_t& elements,
                std::size_t& flops) {
  outputs = shape.c_size();
  if (outputs > std::numeric_limits<std::size_t>::max() / shape.k() / 2) {
    return false;
  }
  flops = outputs * shape.k() * 2;
  const std::size_t a_size = shape.a_size();
  const std::size_t b_size = shape.b_size();
  if (a_size > std::numeric_limits<std::size_t>::max() - b_size) {
    return false;
  }
  const std::size_t inputs = a_size + b_size;
  if (inputs > std::numeric_limits<std::size_t>::max() - outputs) {
    return false;
  }
  elements = inputs + outputs;
  return elements <= std::numeric_limits<std::size_t>::max() / sizeof(float) &&
         flops <= static_cast<std::size_t>(std::numeric_limits<nvbench::int64_t>::max());
}

void add_shape(nvbench::state& state, const GemmShape& shape) {
  add_summary(state, "matmul/height", "Height", shape.m());
  add_summary(state, "matmul/width", "Width", shape.n());
  add_summary(state, "matmul/k", "K", shape.k());
}

void run_benchmark(nvbench::state& state, DeviceData& data, Kernel kernel) {
  if (kernel.launch == nullptr) {
    state.skip("kernel callback is null");
    return;
  }
  const auto& device = state.get_device();
  if (!device.has_value()) {
    state.skip("CUDA device is unavailable");
    return;
  }
  if (!CUDA_CHECK(cudaSetDevice(device.value().get_id())) || !CUDA_CHECK(cudaDeviceSynchronize())) {
    state.skip("CUDA setup failed");
    return;
  }

  std::size_t outputs = 0;
  std::size_t elements = 0;
  std::size_t flops = 0;
  if (!get_counts(data.shape, outputs, elements, flops)) {
    state.skip("matmul metric overflow");
    return;
  }

  add_summary(state, "matmul/flops", "FLOPs", static_cast<nvbench::int64_t>(flops));
  state.add_buffer_size(elements * sizeof(float), "matmul/device_memory", "Memory");
  if (kernel.traffic != nullptr) {
    Traffic traffic;
    if (!kernel.traffic(data.shape, traffic)) {
      state.skip("matmul traffic overflow");
      return;
    }
    state.add_global_memory_reads<float>(traffic.global_reads);
    state.add_global_memory_writes<float>(traffic.global_writes);
  }
#ifdef __clang_analyzer__
  data.c.data()[0] = 0.0F;
  kernel.launch(data.a.data(), data.b.data(), data.c.data(), data.shape, nullptr);
#else
  state.exec(nvbench::exec_tag::gpu, [&](nvbench::launch& launch) {
    kernel.launch(data.a.data(), data.b.data(), data.c.data(), data.shape, launch.get_stream());
  });
  finish_summaries(state, flops);
#endif
}

} // namespace

struct Benchmark::Impl {
  explicit Impl(Problem problem) : host(std::move(problem)) {
  }

  DeviceData& get(nvbench::state& state) {
    const auto& selected_device = state.get_device();
    if (!selected_device.has_value()) {
      throw std::runtime_error("CUDA device is unavailable");
    }
    const int device = selected_device.value().get_id();
    auto found = devices.find(device);
    if (found == devices.end()) {
      if (!CUDA_CHECK(cudaSetDevice(device))) {
        throw std::runtime_error("CUDA device selection failed");
      }
      found = devices.emplace(device, std::make_unique<DeviceData>(host, device)).first;
    }
    return *found->second;
  }

  ~Impl() {
    for (auto& [device, data] : devices) {
      CUDA_CHECK(cudaSetDevice(device));
      data.reset();
    }
  }

  Problem host;
  std::map<int, std::unique_ptr<DeviceData>> devices;
};

Benchmark::Benchmark(Problem problem) : impl_(std::make_unique<Impl>(std::move(problem))) {
}

Benchmark::~Benchmark() = default;

Benchmark::Benchmark(Benchmark&&) noexcept = default;

Benchmark& Benchmark::operator=(Benchmark&&) noexcept = default;

void Benchmark::run(nvbench::state& state, Kernel kernel) {
  run_benchmark(state, impl_->get(state), kernel);
}

void Benchmark::run_with_shape(nvbench::state& state, Kernel kernel) {
  auto& data = impl_->get(state);
  add_shape(state, data.shape);
  run_benchmark(state, data, kernel);
}

void benchmark(nvbench::state& state, GemmShape shape, Kernel kernel) {
  Problem problem(shape);
  fill_random(problem);
  Benchmark harness(std::move(problem));
  harness.run(state, kernel);
}

std::optional<GemmShape> get_shape(nvbench::state& state) {
  const auto height = state.get_int64("Height");
  const auto width = state.get_int64("Width");
  const auto k = state.get_int64("K");
  constexpr auto maximum = static_cast<nvbench::int64_t>(std::numeric_limits<int>::max());
  if (height <= 0 || width <= 0 || k <= 0 || height > maximum || width > maximum || k > maximum) {
    state.skip("matrix dimensions must fit positive unsigned values");
    return std::nullopt;
  }
  return GemmShape(static_cast<unsigned>(height), static_cast<unsigned>(width),
                   static_cast<unsigned>(k));
}

inline int run_nvbench_impl(int argc, char** argv) try { NVBENCH_MAIN_BODY(argc, argv); }
NVBENCH_MAIN_CATCH_EXCEPTIONS

int run_nvbench_args(int argc, char** argv) {
  return run_nvbench_impl(argc, argv);
}

int run_app(int argc, char** argv, const AppConfig& config) try {
  AppArgs args;
  if (!parse_args(argc, argv, args) || (!args.bench && args.remaining.size() != 1)) {
    print_app_usage(argv[0], config.bench_shape);
    return 2;
  }
  if (args.help) {
    print_app_usage(argv[0], config.bench_shape);
    return 0;
  }
  if (!start()) {
    return 1;
  }

  const GemmShape check_shape =
      args.bench ? GemmShape(kDefaultHeight, kDefaultWidth, kDefaultK) : args.shape();
  Problem check_problem(check_shape);
  fill_random(check_problem);
  const int result = check(check_problem, config.kernel);
  if (result != 0 || !args.bench) {
    return result;
  }

  apply_bench_shape(args, config.bench_shape);
  Problem bench_problem(args.shape());
  fill_random(bench_problem);
  Benchmark harness(std::move(bench_problem));
  add_fixed(harness, config.baseline);
  add_fixed(harness, config.kernel);
  HOST_LOG("Benchmark shape: height %u, width %u, k %u", args.height, args.width, args.k);
  return run_nvbench_args(static_cast<int>(args.remaining.size()), args.remaining.data());
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}

int run_app(int argc, char** argv, std::initializer_list<KernelChoice> choices) try {
  if (choices.size() == 0) {
    throw std::invalid_argument("kernel choices are empty");
  }

  const KernelChoice* selected = choices.begin();
  std::vector<char*> common_args;
  if (!parse_choice_args(argc, argv, choices, selected, common_args)) {
    print_choice_usage(argv[0], choices);
    return 2;
  }

  AppArgs args;
  if (!parse_args(static_cast<int>(common_args.size()), common_args.data(), args) ||
      (!args.bench && args.remaining.size() != 1)) {
    print_choice_usage(argv[0], choices);
    return 2;
  }
  if (args.help) {
    print_choice_usage(argv[0], choices);
    return 0;
  }
  if (!start()) {
    return 1;
  }

  Problem problem(args.shape());
  fill_random(problem);
  const int result = check(problem, selected->kernel);
  if (result != 0 || !args.bench) {
    return result;
  }

  for (const auto& choice : choices) {
    add_shaped(choice.kernel);
  }
  static char benchmark_option[] = "--benchmark";
  std::string benchmark_name(selected->kernel.name);
  args.remaining.push_back(benchmark_option);
  args.remaining.push_back(benchmark_name.data());
  return run_nvbench_args(static_cast<int>(args.remaining.size()), args.remaining.data());
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}

int run_bench_app(int argc, char** argv, std::initializer_list<Kernel> kernels) try {
  if (kernels.size() == 0) {
    throw std::invalid_argument("benchmark kernels are empty");
  }

  AppArgs args;
  if (!parse_args(argc, argv, args)) {
    print_bench_usage(argv[0]);
    return 2;
  }
  if (args.help) {
    print_bench_usage(argv[0]);
    static char help[] = "--help";
    args.remaining.push_back(help);
  }
  if (!start()) {
    return 1;
  }

  Problem problem(args.shape());
  fill_random(problem);
  Benchmark harness(std::move(problem));
  for (const auto kernel : kernels) {
    add_fixed(harness, kernel);
  }
  HOST_LOG("Benchmark shape: height %u, width %u, k %u", args.height, args.width, args.k);
  return run_nvbench_args(static_cast<int>(args.remaining.size()), args.remaining.data());
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}

} // namespace lg::matmul
