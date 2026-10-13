#include "benchmark/app.hpp"
#include "benchmark/runner.hpp"
#include "checks.hpp"
#include "log.hpp"
#include "matmul/app.hpp"
#include "matmul/benchmark.hpp"
#include "matmul/correctness.hpp"

#include <algorithm>
#include <cstdio>
#include <cuda/buffer>
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
    harness->run(state, kernel);
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

void print_choice_usage(
  const char* app, std::initializer_list<KernelChoice> choices, const GemmShape& bench_shape
) {
  std::printf(
    "Usage: %s [--kernel %s] [--height N] [--width N] [--k N] [--bench [options]]\n", app,
    join_options(choices).c_str()
  );
  print_defaults("Check defaults", default_shape());
  print_defaults("Bench defaults", bench_shape);
}

const KernelChoice*
find_choice(std::initializer_list<KernelChoice> choices, std::string_view name) {
  const auto* found = std::find_if(choices.begin(), choices.end(), [name](const auto& choice) {
    return choice.option == name;
  });
  return found == choices.end() ? nullptr : found;
}

/// Strips `--kernel <name>` / `--kernel=<name>`; everything else lands in `remaining`.
bool parse_choice_args(
  int argc,
  char** argv,
  std::initializer_list<KernelChoice> choices,
  const KernelChoice*& selected,
  std::vector<char*>& remaining
) {
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

  DeviceData(const Problem& problem, InputTransforms transforms)
      : shape(problem.shape), inputs(problem, transforms),
        a(default_stream(), device_pool(), inputs.a()),
        b(default_stream(), device_pool(), inputs.b()),
        c(default_stream(), device_pool(), problem.shape.c_size(), cuda::no_init) {
  }
};

/// A GEMM does one multiply and one add per (output, k) pair.
std::optional<benchmark::Workload> get_workload(const GemmShape& shape, Kernel kernel) {
  return benchmark::with_no_reuse(
    benchmark::float_workload(
      {{"M", shape.m()}, {"N", shape.n()}, {"K", shape.k()}},
      checked_mul(shape.c_size(), std::size_t{shape.k()} * 2),
      checked_add(shape.a_size(), shape.b_size()), shape.c_size()
    ),
    kernel.traffic, shape
  );
}

void run_benchmark(nvbench::state& state, DeviceData& data, Kernel kernel) {
  if (kernel.launch == nullptr) {
    state.skip("kernel callback is null");
    return;
  }
  benchmark::measure(
    state, get_workload(data.shape, kernel), kernel.resources, data.shape,
    [&](cudaStream_t stream) {
      kernel.launch(data.a.data(), data.b.data(), data.c.data(), data.shape, stream);
    }
  );
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

int launch_nvbench(std::vector<char*>& argv) {
  return benchmark::run_args(static_cast<int>(argv.size()), argv.data());
}

/// Correctness runs on the small default shape; --bench then sizes up.
int check_app(const AppArgs& args, Kernel kernel) {
  Problem problem(args.bench ? default_shape() : args.shape());
  fill_random(problem);
  return check(problem, kernel);
}

void log_shape(const GemmShape& shape) {
  HOST_LOG("Benchmark shape: height %u, width %u, k %u", shape.m(), shape.n(), shape.k());
}

} // namespace

struct Benchmark::Impl {
  explicit Impl(Problem problem) : host(std::move(problem)) {
  }

  /// Runs the reference once; every kernel compares against it.
  bool check(Kernel kernel) {
    if (!reference_ready) {
      reference_ready = run_reference(host);
    }
    return reference_ready && run_kernel(host, kernel) && verify(host);
  }

  Problem host;
  bool reference_ready = false;
  benchmark::DeviceCache<DeviceData, InputTransforms> devices;
};

Benchmark::Benchmark(Problem problem) : impl_(std::make_unique<Impl>(std::move(problem))) {
}

Benchmark::~Benchmark() = default;

Benchmark::Benchmark(Benchmark&&) noexcept = default;

Benchmark& Benchmark::operator=(Benchmark&&) noexcept = default;

void Benchmark::run(nvbench::state& state, Kernel kernel) {
  const auto& device = state.get_device();
  if (!device || !CUDA_CHECK(cudaSetDevice(device->get_id())) || !impl_->check(kernel)) {
    state.skip("benchmark correctness failed");
    return;
  }
  auto& data = impl_->devices.get(device->get_id(), kernel.inputs, impl_->host, kernel.inputs);
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
  return GemmShape(
    static_cast<unsigned>(height), static_cast<unsigned>(width), static_cast<unsigned>(k)
  );
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

    const int result = check_app(args, config.kernel);
    if (result != 0 || !args.bench) {
      return result;
    }

    const GemmShape bench_shape = args.shape(config.bench_shape);
    Problem bench_problem(bench_shape);
    fill_random(bench_problem);
    Benchmark harness(std::move(bench_problem));
    if (config.baseline) {
      add_fixed(harness, *config.baseline);
    }
    add_fixed(harness, config.kernel);
    log_shape(bench_shape);
    return launch_nvbench(args.remaining);
  });
}

int run_app(
  int argc, char** argv, std::initializer_list<KernelChoice> choices, const GemmShape& bench_shape
) {
  return guarded([&] {
    if (choices.size() == 0) {
      throw std::invalid_argument("kernel choices are empty");
    }

    const KernelChoice* selected = choices.begin();
    std::vector<char*> common_args;
    const auto usage = [&] {
      print_choice_usage(argv[0], choices, bench_shape);
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

    const int result = check_app(args, selected->kernel);
    if (result != 0 || !args.bench) {
      return result;
    }

    const GemmShape shape = args.shape(bench_shape);
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
