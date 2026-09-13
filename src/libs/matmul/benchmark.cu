#include "checks.hpp"
#include "matmul/benchmark.hpp"

#include <cuda/buffer>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace lg::matmul {
namespace {

using DeviceMatrix = cuda::device_buffer<float>;

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
  state.add_global_memory_reads<float>(flops);
  state.add_global_memory_writes<float>(outputs);
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

} // namespace lg::matmul
