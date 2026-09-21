#include "benchmark/runner.hpp"
#include "checks.hpp"
#include "convolution3d/benchmark.hpp"
#include "convolution3d/correctness.hpp"

#include <cstdio>
#include <cuda/buffer>
#include <limits>
#include <memory>
#include <nvbench/type_list.cuh>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace lg::convolution3d {
namespace {

using DeviceBuffer = cuda::device_buffer<float>;

struct Counts {
  std::size_t device_bytes;
  std::size_t flops;
};

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

std::optional<Counts> get_counts(Shape shape) {
  const auto xy = checked_mul(
    valid_axis_pairs(shape.width, shape.radius), valid_axis_pairs(shape.height, shape.radius)
  );
  const auto taps =
    xy ? checked_mul(*xy, valid_axis_pairs(shape.depth, shape.radius)) : std::nullopt;
  const auto flops = taps ? checked_mul(*taps, std::size_t{2}) : std::nullopt;
  const auto elements = checked_mul(shape.input_size(), std::size_t{2});
  if (!flops || !elements || *elements > static_cast<std::size_t>(-1) - shape.filter_size()) {
    return std::nullopt;
  }
  const auto bytes = checked_mul(*elements + shape.filter_size(), sizeof(float));
  if (!bytes || *flops > static_cast<std::size_t>(std::numeric_limits<nvbench::int64_t>::max())) {
    return std::nullopt;
  }
  return Counts{*bytes, *flops};
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
    if (!data_ || device_ != device) {
      if (!CUDA_CHECK(cudaSetDevice(device))) {
        state.skip("CUDA device selection failed");
        return;
      }
      data_ = std::make_unique<DeviceData>(host_, device);
      device_ = device;
    }
    const auto counts = get_counts(data_->shape);
    if (!counts) {
      state.skip("convolution metric overflow");
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
    lg::benchmark::add_summary(state, "convolution3d/width", "Width", data_->shape.width);
    lg::benchmark::add_summary(state, "convolution3d/height", "Height", data_->shape.height);
    lg::benchmark::add_summary(state, "convolution3d/depth", "Depth", data_->shape.depth);
    lg::benchmark::add_summary(state, "convolution3d/radius", "Radius", data_->shape.radius);
    lg::benchmark::add_summary(
      state, "convolution3d/flops", "FLOPs", static_cast<nvbench::int64_t>(counts->flops), "flops"
    );
    state.add_buffer_size(counts->device_bytes, "convolution3d/device_memory", "Memory");
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
    lg::benchmark::finish_summaries(state, "convolution3d", counts->flops);
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

int run_app(int argc, char** argv, Kernel kernel, Shape bench_shape) {
  return run_host(argc, argv, [&] {
    AppArgs args;
    if (!parse_args(argc, argv, args)) {
      print_usage(argv[0], bench_shape);
      return 2;
    }
    if (args.help) {
      print_usage(argv[0], bench_shape);
      return 0;
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
  });
}

} // namespace lg::convolution3d
