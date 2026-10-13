#include "benchmark/runner.hpp"
#include "checks.hpp"
#include "stencil/benchmark.hpp"
#include "stencil/correctness.hpp"

#include <cstdio>
#include <cuda/buffer>
#include <memory>
#include <nvbench/type_list.cuh>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace lg::stencil {
namespace {

using DeviceBuffer = cuda::device_buffer<float>;

// 7 multiplies and 6 adds per interior point.
constexpr std::size_t kFlopsPerPoint = 13;

struct AppArgs {
  bool bench = false;
  bool help = false;
  std::vector<char*> remaining;
};

struct DeviceData {
  Shape shape;
  Coefficients coefficients;
  DeviceBuffer input;
  DeviceBuffer output;

  DeviceData(const Problem& problem, int device)
      : DeviceData(problem, cuda::device_default_memory_pool(cuda::devices[device])) {
  }

private:
  template <typename Pool>
  DeviceData(const Problem& problem, Pool& pool)
      : shape(problem.shape), coefficients(problem.coefficients),
        input(default_stream(), pool, problem.input),
        output(default_stream(), pool, initial_output(problem)) {
  }
};

std::optional<lg::benchmark::Workload> get_workload(Shape shape, Kernel kernel) {
  const auto flops = checked_mul(shape.interior_size(), kFlopsPerPoint);
  const auto buffers = checked_mul(shape.grid_size(), 2 * sizeof(float));
  const auto minimum = lg::benchmark::float_traffic(shape.grid_size(), shape.interior_size());
  if (!flops || !buffers || !minimum) {
    return std::nullopt;
  }
  lg::benchmark::Workload workload{
    {{"W", shape.width}, {"H", shape.height}, {"D", shape.depth}}, *flops, *buffers, *minimum, {}};
  if (kernel.traffic != nullptr) {
    Traffic traffic;
    if (!kernel.traffic(shape, traffic)) {
      return std::nullopt;
    }
    workload.no_reuse = lg::benchmark::float_traffic(traffic.global_reads, traffic.global_writes);
    if (!workload.no_reuse) {
      return std::nullopt;
    }
  }
  return workload;
}

class Benchmark {
public:
  explicit Benchmark(Problem problem) : host_(std::move(problem)) {
  }

  void run(nvbench::state& state, Kernel kernel) {
    const auto& selected = state.get_device();
    if (!selected.has_value()) {
      state.skip("CUDA device is unavailable");
      return;
    }
    const int device = selected.value().get_id();
    // run_app already compared every value on the small shape.
    constexpr std::size_t reference_samples = 257;
    if (!CUDA_CHECK(cudaSetDevice(device)) || check(host_, kernel, reference_samples) != 0) {
      state.skip("benchmark correctness failed");
      return;
    }
    if (!data_ || device_ != device) {
      data_ = std::make_unique<DeviceData>(host_, device);
      device_ = device;
    }
    if (!CUDA_CHECK(cudaDeviceSynchronize())) {
      state.skip("CUDA setup failed");
      return;
    }
    const auto workload = get_workload(data_->shape, kernel);
    if (!workload) {
      state.skip("stencil workload overflow");
      return;
    }
    if (kernel.resources != nullptr) {
      lg::benchmark::KernelResources resources;
      if (!CUDA_CHECK(kernel.resources(data_->shape, resources))) {
        state.skip("CUDA kernel resource query failed");
        return;
      }
      lg::benchmark::add_resources(state, resources);
    }
    if (!lg::benchmark::describe(state, *workload)) {
      return;
    }
#ifdef __clang_analyzer__
    data_->output.data()[0] = 0.0F;
    kernel.launch(
      data_->input.data(), data_->output.data(), data_->shape, data_->coefficients, nullptr
    );
#else
    state.exec(nvbench::exec_tag::gpu, [&](nvbench::launch& launch) {
      kernel.launch(
        data_->input.data(), data_->output.data(), data_->shape, data_->coefficients,
        launch.get_stream()
      );
    });
    lg::benchmark::finish_summaries(state, *workload);
#endif
  }

private:
  Problem host_;
  int device_ = -1;
  std::unique_ptr<DeviceData> data_;
};

struct Runner {
  Benchmark* benchmark;
  Kernel kernel;

  void operator()(nvbench::state& state, nvbench::type_list<>) const {
    benchmark->run(state, kernel);
  }
};

bool parse_args(int argc, char** argv, AppArgs& args) {
  args.remaining.push_back(argv[0]);
  for (int index = 1; index < argc; ++index) {
    const std::string_view arg = argv[index];
    if (arg == "--bench") {
      args.bench = true;
    } else if (arg == "--help" || arg == "-h") {
      args.help = true;
    } else {
      args.remaining.push_back(argv[index]);
    }
  }
  return args.bench || args.remaining.size() == 1;
}

void print_usage(const char* app, Shape shape) {
  std::printf("Usage: %s [--bench [NVBench options]]\n", app);
  std::printf("Bench shape: %dx%dx%d\n", shape.width, shape.height, shape.depth);
}

} // namespace

int run_app(int argc, char** argv, Kernel kernel, Shape bench_shape) try {
  AppArgs args;
  if (!parse_args(argc, argv, args)) {
    print_usage(argv[0], bench_shape);
    return 2;
  }
  if (args.help) {
    print_usage(argv[0], bench_shape);
    return 0;
  }
  if (!init_host()) {
    return 1;
  }

  Problem problem{kDefaultShape, {}, {}, {}, {}};
  fill_problem(problem);
  if (const int result = check(problem, kernel); result != 0) {
    return result;
  }
  if (!args.bench) {
    return 0;
  }

  Problem bench_problem{bench_shape, {}, {}, {}, {}};
  fill_problem(bench_problem);
  Benchmark benchmark(std::move(bench_problem));
  lg::benchmark::add(kernel.name, Runner{&benchmark, kernel});
  return lg::benchmark::run_args(static_cast<int>(args.remaining.size()), args.remaining.data());
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}

} // namespace lg::stencil
