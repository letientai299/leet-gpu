#include "benchmark/runner.hpp"
#include "checks.hpp"
#include "log.hpp"
#include "matmul/app.hpp"
#include "matmul/benchmark.hpp"
#include "matmul/correctness.hpp"

#include <algorithm>
#include <cstdio>
#include <cuda/buffer>
#include <iterator>
#include <memory>
#include <nvbench/type_list.cuh>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lg::matmul {
namespace {

using DeviceMatrix = cuda::device_buffer<float>;

/// Reuses the device copies held by `harness` across states.
struct FixedRunner {
  Benchmark* harness;
  Kernel kernel;

  void operator()(nvbench::state& state, nvbench::type_list<>) const {
    harness->run_with_shape(state, kernel);
  }
};

/// Rebuilds the problem per state so NVBench can sweep the shape axes.
struct ShapeRunner {
  Kernel kernel;

  void operator()(nvbench::state& state, nvbench::type_list<>) const {
    if (const auto shape = get_shape(state)) {
      benchmark(state, *shape, kernel);
    }
  }
};

template <typename Runner> nvbench::benchmark_base& add_benchmark(Kernel kernel, Runner runner) {
  return benchmark::add(kernel.name, std::move(runner));
}

void add_fixed(Benchmark& harness, Kernel kernel) {
  add_benchmark(kernel, FixedRunner{&harness, kernel});
}

void add_shaped(Kernel kernel, const GemmShape& shape) {
  add_benchmark(kernel, ShapeRunner{kernel})
      .add_int64_axis("Height", {shape.m()})
      .add_int64_axis("Width", {shape.n()})
      .add_int64_axis("K", {shape.k()});
}

void print_defaults(const char* label, const GemmShape& shape) {
  std::printf("%s: --height %u --width %u --k %u\n", label, shape.m(), shape.n(), shape.k());
}

void print_app_usage(const char* app, const GemmShape& bench_shape) {
  std::printf("Usage: %s [--height N] [--width N] [--k N] [--bench [options]]\n", app);
  print_defaults("Check defaults", default_shape());
  print_defaults("Bench defaults", bench_shape);
}

std::string join_options(std::initializer_list<KernelChoice> choices) {
  std::string joined;
  for (const auto& choice : choices) {
    if (!joined.empty()) {
      joined.push_back('|');
    }
    joined.append(choice.option);
  }
  return joined;
}

void print_choice_usage(const char* app, std::initializer_list<KernelChoice> choices) {
  std::printf("Usage: %s [--kernel %s] [--height N] [--width N] [--k N] [--bench [options]]\n", app,
              join_options(choices).c_str());
  print_defaults("Defaults", default_shape());
}

const KernelChoice* find_choice(std::initializer_list<KernelChoice> choices,
                                std::string_view name) {
  const auto* found = std::find_if(choices.begin(), choices.end(), [name](const auto& choice) {
    return choice.option == name;
  });
  return found == choices.end() ? nullptr : found;
}

/// Strips `--kernel <name>` / `--kernel=<name>`; everything else lands in `remaining`.
bool parse_choice_args(int argc,
                       char** argv,
                       std::initializer_list<KernelChoice> choices,
                       const KernelChoice*& selected,
                       std::vector<char*>& remaining) {
  constexpr std::string_view flag = "--kernel";
  remaining.push_back(argv[0]);
  for (int index = 1; index < argc; ++index) {
    const std::string_view arg = argv[index];
    std::string_view name;
    if (arg == flag) {
      if (++index == argc) {
        return false;
      }
      name = argv[index];
    } else if (arg.size() > flag.size() && arg.substr(0, flag.size()) == flag &&
               arg[flag.size()] == '=') {
      name = arg.substr(flag.size() + 1);
    } else {
      remaining.push_back(argv[index]);
      continue;
    }
    selected = find_choice(choices, name);
    if (selected == nullptr) {
      return false;
    }
  }
  return true;
}

struct DeviceData {
  GemmShape shape;
  KernelInputs inputs;
  DeviceMatrix a;
  DeviceMatrix b;
  DeviceMatrix c;

  DeviceData(const Problem& problem, InputTransforms transforms, int device)
      : DeviceData(problem, transforms, cuda::device_default_memory_pool(cuda::devices[device])) {
  }

private:
  template <typename Pool>
  DeviceData(const Problem& problem, InputTransforms transforms, Pool& pool)
      : shape(problem.shape), inputs(problem, transforms), a(default_stream(), pool, inputs.a()),
        b(default_stream(), pool, inputs.b()), c(default_stream(), pool, problem.result) {
  }
};

struct Counts {
  std::size_t elements = 0;
  std::size_t flops = 0;
};

/// A GEMM does one multiply and one add per (output, k) pair.
std::optional<Counts> get_counts(const GemmShape& shape) {
  const auto flops = checked_mul(shape.c_size(), std::size_t{shape.k()} * 2);
  const auto inputs = checked_add(shape.a_size(), shape.b_size());
  if (!flops || !inputs) {
    return std::nullopt;
  }
  const auto elements = checked_add(*inputs, shape.c_size());
  if (!elements || *elements > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      *flops > static_cast<std::size_t>(std::numeric_limits<nvbench::int64_t>::max())) {
    return std::nullopt;
  }
  return Counts{*elements, *flops};
}

/// PMPP counts the traffic a cache-less machine would move, which is the whole
/// point of tiling. It is NOT DRAM traffic, so it must not go to
/// add_global_memory_reads: NVBench divides that by DRAM peak and reported 456%
/// bandwidth utilisation. Reported as a model instead, so the column never
/// claims to be measured. https://en.wikipedia.org/wiki/Roofline_model
void add_model_traffic(nvbench::state& state, const Traffic& traffic, std::size_t flops) {
  const auto elements = checked_add(traffic.global_reads, traffic.global_writes);
  if (!elements) {
    return;
  }
  const auto bytes = checked_mul(*elements, sizeof(float));
  if (!bytes || *bytes == 0) {
    return;
  }
  benchmark::add_summary(state, "matmul/model/bytes", "Model Bytes",
                         static_cast<nvbench::int64_t>(*bytes), "bytes");
  auto& summary = state.add_summary("matmul/model/intensity");
  summary.set_string("name", "Model FLOP/B");
  summary.set_string("description", "Arithmetic intensity of the cache-less traffic model");
  summary.set_float64("value", static_cast<double>(flops) / static_cast<double>(*bytes));
}

void add_shape(nvbench::state& state, const GemmShape& shape) {
  benchmark::add_summary(state, "matmul/height", "Height", shape.m());
  benchmark::add_summary(state, "matmul/width", "Width", shape.n());
  benchmark::add_summary(state, "matmul/k", "K", shape.k());
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

  const auto counts = get_counts(data.shape);
  if (!counts) {
    state.skip("matmul metric overflow");
    return;
  }

  benchmark::add_summary(state, "matmul/flops", "FLOPs",
                         static_cast<nvbench::int64_t>(counts->flops), "flops");
  state.add_buffer_size(counts->elements * sizeof(float), "matmul/device_memory", "Memory");
  if (kernel.traffic != nullptr) {
    Traffic traffic;
    if (!kernel.traffic(data.shape, traffic)) {
      state.skip("matmul traffic overflow");
      return;
    }
    add_model_traffic(state, traffic, counts->flops);
  }
#ifdef __clang_analyzer__
  data.c.data()[0] = 0.0F;
  kernel.launch(data.a.data(), data.b.data(), data.c.data(), data.shape, nullptr);
#else
  state.exec(nvbench::exec_tag::gpu, [&](nvbench::launch& launch) {
    kernel.launch(data.a.data(), data.b.data(), data.c.data(), data.shape, launch.get_stream());
  });
  benchmark::finish_summaries(state, "matmul", counts->flops);
#endif
}

/// Shared prologue: parse, honour --help, then bring up the host.
/// Returns an exit code when the app should stop, nullopt to keep going.
template <typename PrintUsage>
std::optional<int> prepare(int argc, char** argv, AppArgs& args, PrintUsage print_usage) {
  if (!parse_args(argc, argv, args)) {
    print_usage();
    return 2;
  }
  if (args.help) {
    print_usage();
    return 0;
  }
  // Outside --bench there is no NVBench to consume leftovers, so they are typos.
  if (!args.bench && args.remaining.size() != 1) {
    print_usage();
    return 2;
  }
  return start() ? std::nullopt : std::optional<int>(1);
}

template <typename Body> int guarded(Body&& body) try {
  return body();
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}

int launch_nvbench(std::vector<char*>& argv) {
  return benchmark::run_args(static_cast<int>(argv.size()), argv.data());
}

void log_shape(const GemmShape& shape) {
  HOST_LOG("Benchmark shape: height %u, width %u, k %u", shape.m(), shape.n(), shape.k());
}

} // namespace

struct Benchmark::Impl {
  struct DeviceEntry {
    int device;
    InputTransforms inputs;
    std::unique_ptr<DeviceData> data;
  };

  explicit Impl(Problem problem) : host(std::move(problem)) {
  }

  ~Impl() {
    for (auto& entry : devices) {
      CUDA_CHECK(cudaSetDevice(entry.device));
      entry.data.reset();
    }
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;

  /// Uploads once per device, then hands the same buffers to every state.
  DeviceData& get(nvbench::state& state, Kernel kernel) {
    const auto& selected_device = state.get_device();
    if (!selected_device.has_value()) {
      throw std::runtime_error("CUDA device is unavailable");
    }
    const int device = selected_device.value().get_id();
    auto found = std::find_if(devices.begin(), devices.end(), [&](const auto& entry) {
      return entry.device == device && entry.inputs == kernel.inputs;
    });
    if (found == devices.end()) {
      if (!CUDA_CHECK(cudaSetDevice(device))) {
        throw std::runtime_error("CUDA device selection failed");
      }
      devices.push_back(DeviceEntry{device, kernel.inputs,
                                    std::make_unique<DeviceData>(host, kernel.inputs, device)});
      found = std::prev(devices.end());
    }
    return *found->data;
  }

  Problem host;
  std::vector<DeviceEntry> devices;
};

Benchmark::Benchmark(Problem problem) : impl_(std::make_unique<Impl>(std::move(problem))) {
}

Benchmark::~Benchmark() = default;

Benchmark::Benchmark(Benchmark&&) noexcept = default;

Benchmark& Benchmark::operator=(Benchmark&&) noexcept = default;

void Benchmark::run(nvbench::state& state, Kernel kernel) {
  run_benchmark(state, impl_->get(state, kernel), kernel);
}

void Benchmark::run_with_shape(nvbench::state& state, Kernel kernel) {
  auto& data = impl_->get(state, kernel);
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
  const auto in_range = [](nvbench::int64_t value) {
    return value > 0 && value <= static_cast<nvbench::int64_t>(kMaxDimension);
  };
  if (!in_range(height) || !in_range(width) || !in_range(k)) {
    state.skip("matrix dimensions must fit positive unsigned values");
    return std::nullopt;
  }
  return GemmShape(static_cast<unsigned>(height), static_cast<unsigned>(width),
                   static_cast<unsigned>(k));
}

int run_nvbench_args(int argc, char** argv) {
  return benchmark::run_args(argc, argv);
}

int run_app(int argc, char** argv, const AppConfig& config) {
  return guarded([&] {
    AppArgs args;
    const auto usage = [&] {
      print_app_usage(argv[0], config.bench_shape);
    };
    if (const auto status = prepare(argc, argv, args, usage)) {
      return *status;
    }

    // Correctness runs on the small default shape; --bench then sizes up.
    Problem check_problem(args.bench ? default_shape() : args.shape());
    fill_random(check_problem);
    const int result = check(check_problem, config.kernel);
    if (result != 0 || !args.bench) {
      return result;
    }

    const GemmShape bench_shape = args.shape(config.bench_shape);
    Problem bench_problem(bench_shape);
    fill_random(bench_problem);
    Benchmark harness(std::move(bench_problem));
    add_fixed(harness, config.baseline);
    add_fixed(harness, config.kernel);
    log_shape(bench_shape);
    return launch_nvbench(args.remaining);
  });
}

int run_app(int argc, char** argv, std::initializer_list<KernelChoice> choices) {
  return guarded([&] {
    if (choices.size() == 0) {
      throw std::invalid_argument("kernel choices are empty");
    }

    const KernelChoice* selected = choices.begin();
    std::vector<char*> common_args;
    const auto usage = [&] {
      print_choice_usage(argv[0], choices);
    };
    if (!parse_choice_args(argc, argv, choices, selected, common_args)) {
      usage();
      return 2;
    }

    AppArgs args;
    if (const auto status =
            prepare(static_cast<int>(common_args.size()), common_args.data(), args, usage)) {
      return *status;
    }

    const GemmShape shape = args.shape();
    Problem problem(shape);
    fill_random(problem);
    const int result = check(problem, selected->kernel);
    if (result != 0 || !args.bench) {
      return result;
    }

    // Only the selected kernel is registered, so NVBench needs no --benchmark filter.
    add_shaped(selected->kernel, shape);
    log_shape(shape);
    return launch_nvbench(args.remaining);
  });
}

int run_bench_app(int argc, char** argv, std::initializer_list<Kernel> kernels) {
  return guarded([&] {
    if (kernels.size() == 0) {
      throw std::invalid_argument("benchmark kernels are empty");
    }

    AppArgs args;
    if (!parse_args(argc, argv, args)) {
      print_bench_usage(argv[0]);
      return 2;
    }
    static char help_flag[] = "--help";
    if (args.help) {
      print_bench_usage(argv[0]);
      args.remaining.push_back(help_flag); // Let NVBench list its own options too.
    }
    if (!start()) {
      return 1;
    }

    const GemmShape shape = args.shape();
    Problem problem(shape);
    fill_random(problem);
    Benchmark harness(std::move(problem));
    for (const auto kernel : kernels) {
      add_fixed(harness, kernel);
    }
    log_shape(shape);
    return launch_nvbench(args.remaining);
  });
}

} // namespace lg::matmul
