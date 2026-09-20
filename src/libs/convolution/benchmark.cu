#include "benchmark/runner.hpp"
#include "checks.hpp"
#include "convolution/app.hpp"
#include "convolution/benchmark.hpp"
#include "convolution/correctness.hpp"
#include "log.hpp"

#include <cstdio>
#include <cuda/buffer>
#include <limits>
#include <memory>
#include <nvbench/type_list.cuh>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace lg::convolution {
namespace {

using DeviceBuffer = cuda::device_buffer<float>;

struct AppArgs {
  bool bench = false;
  bool help = false;
  std::vector<char*> remaining;
};

struct Counts {
  std::size_t device_bytes = 0;
  std::size_t flops = 0;
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
    run_benchmark(state, *data_, kernel);
  }

private:
  static std::optional<Counts> get_counts(Shape shape) {
    const auto taps = checked_mul(
      valid_axis_pairs(shape.width, shape.radius), valid_axis_pairs(shape.height, shape.radius)
    );
    const auto flops = taps ? checked_mul(*taps, std::size_t{2}) : std::nullopt;
    const auto elements = checked_mul(shape.input_size(), std::size_t{2});
    if (!flops || !elements || *elements > static_cast<std::size_t>(-1) - shape.filter_size()) {
      return std::nullopt;
    }
    const auto device_bytes = checked_mul(*elements + shape.filter_size(), sizeof(float));
    if (!device_bytes ||
        *flops > static_cast<std::size_t>(std::numeric_limits<nvbench::int64_t>::max())) {
      return std::nullopt;
    }
    return Counts{*device_bytes, *flops};
  }

  static std::optional<std::size_t>
  add_traffic(nvbench::state& state, Shape shape, Kernel kernel, std::size_t flops) {
    if (kernel.traffic == nullptr) {
      return std::nullopt;
    }
    Traffic traffic;
    if (!kernel.traffic(shape, traffic) ||
        traffic.global_reads > static_cast<std::size_t>(-1) - traffic.global_writes) {
      state.skip("convolution traffic overflow");
      return std::nullopt;
    }
    const auto bytes = checked_mul(traffic.global_reads + traffic.global_writes, sizeof(float));
    if (!bytes || *bytes > static_cast<std::size_t>(std::numeric_limits<nvbench::int64_t>::max())) {
      state.skip("convolution traffic overflow");
      return std::nullopt;
    }
    lg::benchmark::add_summary(
      state, "convolution/model/gmem_bytes", "Model GMEM Bytes",
      static_cast<nvbench::int64_t>(*bytes), "bytes"
    );
    auto& intensity = state.add_summary("convolution/model/intensity");
    intensity.set_string("name", "Model FLOP/B");
    intensity.set_string("description", "Arithmetic intensity of the logical GMEM traffic model");
    intensity.set_float64("value", static_cast<double>(flops) / static_cast<double>(*bytes));
    return bytes;
  }

  static bool add_resources(nvbench::state& state, Shape shape, Kernel kernel) {
    if (kernel.resources == nullptr) {
      return true;
    }
    lg::benchmark::KernelResources resources;
    if (!CUDA_CHECK(kernel.resources(shape, resources))) {
      state.skip("CUDA kernel resource query failed");
      return false;
    }

    lg::benchmark::add_resources(state, resources);
    return true;
  }

  static void run_benchmark(nvbench::state& state, DeviceData& data, Kernel kernel) {
    if (kernel.launch == nullptr) {
      state.skip("kernel callback is null");
      return;
    }
    if (!CUDA_CHECK(cudaDeviceSynchronize())) {
      state.skip("CUDA setup failed");
      return;
    }
    const auto counts = get_counts(data.shape);
    if (!counts) {
      state.skip("convolution metric overflow");
      return;
    }
    if (!add_resources(state, data.shape, kernel)) {
      return;
    }
    const auto model_bytes = add_traffic(state, data.shape, kernel, counts->flops);
    if (kernel.traffic != nullptr && !model_bytes) {
      return;
    }

    lg::benchmark::add_summary(state, "convolution/width", "Width", data.shape.width);
    lg::benchmark::add_summary(state, "convolution/height", "Height", data.shape.height);
    lg::benchmark::add_summary(state, "convolution/radius", "Radius", data.shape.radius);
    lg::benchmark::add_summary(
      state, "convolution/flops", "FLOPs", static_cast<nvbench::int64_t>(counts->flops), "flops"
    );
    state.add_buffer_size(counts->device_bytes, "convolution/device_memory", "Memory");
#ifdef __clang_analyzer__
    data.output.data()[0] = 0.0F;
    kernel.launch(data.input.data(), data.filter.data(), data.output.data(), data.shape, nullptr);
#else
    state.exec(nvbench::exec_tag::gpu, [&](nvbench::launch& launch) {
      kernel.launch(
        data.input.data(), data.filter.data(), data.output.data(), data.shape, launch.get_stream()
      );
    });
    lg::benchmark::finish_summaries(state, "convolution", counts->flops, model_bytes);
#endif
  }

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
  std::printf("Bench shape: %dx%d, radius %d\n", shape.width, shape.height, shape.radius);
}

template <typename Body> int guarded(Body&& body) try {
  return body();
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}

} // namespace

int run_app(int argc, char** argv, Kernel kernel, Shape bench_shape) {
  return guarded([&] {
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

    Problem check_problem{kDefaultShape, {}, {}, {}, {}};
    fill_problem(check_problem);
    const int result = check(check_problem, kernel);
    if (result != 0 || !args.bench) {
      return result;
    }

    Problem bench_problem{bench_shape, {}, {}, {}, {}};
    fill_problem(bench_problem);
    Benchmark benchmark(std::move(bench_problem));
    lg::benchmark::add(kernel.name, Runner{&benchmark, kernel});
    HOST_LOG(
      "Benchmark shape: width %d, height %d, radius %d", bench_shape.width, bench_shape.height,
      bench_shape.radius
    );
    return lg::benchmark::run_args(static_cast<int>(args.remaining.size()), args.remaining.data());
  });
}

} // namespace lg::convolution
