#include "benchmark/runner.hpp"
#include "checks.hpp"
#include "convolution3d/benchmark.hpp"
#include "convolution3d/correctness.hpp"

#include <cstdio>
#include <cuda/buffer>
#include <memory>
#include <nvbench/type_list.cuh>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace lg::convolution3d {
namespace {

using DeviceBuffer = cuda::device_buffer<float>;

struct AppArgs {
  bool bench = false;
  bool help = false;
  std::vector<char*> remaining;
};

struct DeviceData {
  Shape shape;
  DeviceBuffer input;
  DeviceBuffer filter;
  DeviceBuffer output;

  DeviceData(const Problem& problem, int device)
      : DeviceData(problem, cuda::device_default_memory_pool(cuda::devices[device])) {
  }

private:
  template <typename Pool>
  DeviceData(const Problem& problem, Pool& pool)
      : shape(problem.shape), input(default_stream(), pool, problem.input),
        filter(default_stream(), pool, problem.filter),
        output(default_stream(), pool, problem.result) {
  }
};

std::optional<lg::benchmark::Workload> get_workload(Shape shape, Kernel kernel) {
  const auto xy = checked_mul(
    valid_axis_pairs(shape.width, shape.radius), valid_axis_pairs(shape.height, shape.radius)
  );
  const auto taps =
    xy ? checked_mul(*xy, valid_axis_pairs(shape.depth, shape.radius)) : std::nullopt;
  const auto flops = taps ? checked_mul(*taps, std::size_t{2}) : std::nullopt;
  const auto reads = lg::checked_add(shape.input_size(), shape.filter_size());
  const auto minimum =
    reads ? lg::benchmark::float_traffic(*reads, shape.input_size()) : std::nullopt;
  const auto buffers =
    minimum ? lg::checked_add(minimum->read_bytes, minimum->write_bytes) : std::nullopt;
  if (!flops || !minimum || !buffers) {
    return std::nullopt;
  }
  lg::benchmark::Workload workload{
    {{"W", shape.width}, {"H", shape.height}, {"D", shape.depth}, {"r", shape.radius}},
    *flops,
    *buffers,
    *minimum,
    {}};
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
    constexpr std::size_t reference_samples = 257;
    if (!CUDA_CHECK(cudaSetDevice(device)) || check(host_, kernel, reference_samples) != 0) {
      state.skip("benchmark correctness failed");
      return;
    }
    if (!data_ || device_ != device) {
      if (!CUDA_CHECK(cudaSetDevice(device))) {
        state.skip("CUDA device selection failed");
        return;
      }
      data_ = std::make_unique<DeviceData>(host_, device);
      device_ = device;
    }
    const auto workload = get_workload(data_->shape, kernel);
    if (!workload) {
      state.skip("convolution workload overflow");
      return;
    }
    if (kernel.filter_setup != nullptr &&
        !CUDA_CHECK(kernel.filter_setup(data_->filter.data(), data_->shape, nullptr))) {
      state.skip("CUDA filter setup failed");
      return;
    }
    if (!CUDA_CHECK(cudaDeviceSynchronize())) {
      state.skip("CUDA setup failed");
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
      data_->input.data(), data_->filter.data(), data_->output.data(), data_->shape, nullptr
    );
#else
    state.exec(nvbench::exec_tag::gpu, [&](nvbench::launch& launch) {
      kernel.launch(
        data_->input.data(), data_->filter.data(), data_->output.data(), data_->shape,
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
  std::printf(
    "Bench shape: %dx%dx%d, radius %d\n", shape.width, shape.height, shape.depth, shape.radius
  );
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

} // namespace lg::convolution3d
