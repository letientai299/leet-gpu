#pragma once

#include "checks.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cuda/buffer>
#include <cuda_runtime_api.h>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#ifdef LEET_GPU_HAS_NVBENCH
#include <nvbench/main.cuh>
#include <nvbench/nvbench.cuh>
#endif

using dbuf = cuda::device_buffer<float>;
using MatmulKernel =
    void (*)(const float*, const float*, float*, unsigned, unsigned, unsigned, cudaStream_t);

void launch_matmul_cell(const float* a,
                        const float* b,
                        float* c,
                        unsigned height,
                        unsigned width,
                        unsigned k,
                        cudaStream_t stream = nullptr);
void launch_matmul_row(const float* a,
                       const float* b,
                       float* c,
                       unsigned height,
                       unsigned width,
                       unsigned k,
                       cudaStream_t stream = nullptr);
void launch_matmul_col(const float* a,
                       const float* b,
                       float* c,
                       unsigned height,
                       unsigned width,
                       unsigned k,
                       cudaStream_t stream = nullptr);

// Host driver: random A/B, CUTLASS oracle, then the kernel under test.
// C[height, width] = A[height, k] * B[k, width], all row-major.
struct Matmul {
  unsigned height;             // rows of A and C
  unsigned width;              // cols of B and C
  unsigned k;                  // cols of A / rows of B
  std::vector<float> a;        // row-major height*k
  std::vector<float> b;        // row-major k*width
  std::vector<float> expected; // CUTLASS C
  std::vector<float> c;        // kernel C

  /// Allocate host A, B, C, and expected for C = A * B.
  Matmul(unsigned height, unsigned width, unsigned inner)
      : height(height), width(width), k(inner), a(static_cast<std::size_t>(height) * k),
        b(static_cast<std::size_t>(k) * width), expected(static_cast<std::size_t>(height) * width),
        c(expected.size()) {
  }

  /// Fill A and B with random values in [-1, 1].
  void fill();
  /// Write CUTLASS C into expected. Does not call the kernel under test.
  bool reference();
  /// True if c matches expected within atol/rtol.
  [[nodiscard]] bool verify() const;

  /// Copy A/B to the device, run op on C, copy C into out.
  template <typename Op> bool on_device(std::vector<float>& out, Op&& op) {
    const auto stream = default_stream();
    auto& pool = device_pool();
    const dbuf dev_a{stream, pool, a};
    const dbuf dev_b{stream, pool, b};
    // Copy host C (zeros). no_init reuses the pool block the oracle just wrote,
    // so an empty kernel would memcpy that leftover GEMM and pass verify.
    dbuf dev_c{stream, pool, out};
    if (!op(dev_a, dev_b, dev_c)) {
      return false;
    }
    const auto bytes = out.size() * sizeof(float);
    return CUDA_CHECK(cudaMemcpy(out.data(), dev_c.data(), bytes, cudaMemcpyDeviceToHost));
  }

  /// Fill A/B and run the CUTLASS oracle.
  bool init() {
    fill();
    return reference();
  }

  /// Launch the kernel under test and write C into c.
  template <typename Launch> bool multiply_on_gpu(Launch&& launch) {
    return on_device(c, [&](const dbuf& dev_a, const dbuf& dev_b, dbuf& dev_c) {
      return launch(dev_a, dev_b, dev_c, height, width, k);
    });
  }

  /// Run oracle, kernel, and verify. Returns 0 on success.
  template <typename Launch> int run(Launch&& launch) {
    return init() && multiply_on_gpu(std::forward<Launch>(launch)) && verify() ? 0 : 1;
  }
};

inline int run_matmul_kernel(unsigned height, unsigned width, unsigned k, MatmulKernel launch) {
  return Matmul(height, width, k)
      .run([launch](const dbuf& a, const dbuf& b, dbuf& c, unsigned height, unsigned width,
                    unsigned k) {
        launch(a.data(), b.data(), c.data(), height, width, k, nullptr);
        return CUDA_CHECK(cudaGetLastError());
      });
}

inline void Matmul::fill() {
  // Stable inputs make correctness and benchmark runs reproducible.
  std::mt19937 rng(0x4D41544DU); // NOLINT(bugprone-random-generator-seed)
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  const auto next = [&] {
    return dist(rng);
  };
  std::generate(a.begin(), a.end(), next);
  std::generate(b.begin(), b.end(), next);
}

inline bool Matmul::verify() const {
  // Naive GEMM and CUTLASS need not match bitwise (FMA, add order).
  constexpr float atol = 1.0e-5F;
  constexpr float rtol = 1.0e-4F;
  for (std::size_t index = 0; index < c.size(); ++index) {
    const float diff = std::fabs(c[index] - expected[index]);
    const float tol = atol + rtol * std::fabs(expected[index]);
    if (diff > tol) {
      HOST_LOG("Mismatch at %zu: %g vs %g (diff %g, tol %g)", index, c[index], expected[index],
               diff, tol);
      return false;
    }
  }
  HOST_LOG("Matrix multiply passed: %zu values", c.size());
  return true;
}

#ifdef LEET_GPU_HAS_NVBENCH
struct MatmulDeviceData {
  unsigned height;
  unsigned width;
  unsigned k;
  dbuf a;
  dbuf b;
  dbuf c;

  MatmulDeviceData(const Matmul& matmul, int device)
      : height(matmul.height), width(matmul.width), k(matmul.k),
        a(default_stream(), cuda::device_default_memory_pool(cuda::devices[device]), matmul.a),
        b(default_stream(), cuda::device_default_memory_pool(cuda::devices[device]), matmul.b),
        c(default_stream(), cuda::device_default_memory_pool(cuda::devices[device]), matmul.c) {
  }
};

inline void add_matmul_summary(nvbench::state& state,
                               std::string tag,
                               std::string name,
                               nvbench::int64_t value) {
  auto& summary = state.add_summary(std::move(tag));
  summary.set_string("name", std::move(name));
  summary.set_int64("value", value);
}

inline void finish_matmul_summaries(nvbench::state& state, std::size_t flops) {
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

inline void benchmark_matmul(nvbench::state& state,
                             MatmulDeviceData& data,
                             MatmulKernel launch,
                             bool report_dimensions = false) {
  const auto& device = state.get_device();
  if (!device.has_value()) {
    state.skip("CUDA device is unavailable");
    return;
  }
  if (!CUDA_CHECK(cudaSetDevice(device.value().get_id())) || !CUDA_CHECK(cudaDeviceSynchronize())) {
    state.skip("CUDA setup failed");
    return;
  }

  const auto outputs = static_cast<std::size_t>(data.height) * data.width;
  const auto matrix_elements = static_cast<std::size_t>(data.height) * data.k +
                               static_cast<std::size_t>(data.k) * data.width + outputs;
  if (outputs > std::numeric_limits<std::size_t>::max() / data.k / 2) {
    state.skip("FLOP count overflow");
    return;
  }
  const auto flops = outputs * data.k * 2;
  const auto reads = flops;

  if (report_dimensions) {
    add_matmul_summary(state, "matmul/height", "Height", data.height);
    add_matmul_summary(state, "matmul/width", "Width", data.width);
    add_matmul_summary(state, "matmul/k", "K", data.k);
  }
  add_matmul_summary(state, "matmul/flops", "FLOPs", static_cast<nvbench::int64_t>(flops));
  state.add_buffer_size(matrix_elements * sizeof(float), "matmul/device_memory", "Memory");
  state.add_global_memory_reads<float>(reads);
  state.add_global_memory_writes<float>(outputs);
#ifdef __clang_analyzer__
  data.c.data()[0] = 0.0F;
  launch(data.a.data(), data.b.data(), data.c.data(), data.height, data.width, data.k, nullptr);
#else
  state.exec(nvbench::exec_tag::gpu, [&](nvbench::launch& bench) {
    launch(data.a.data(), data.b.data(), data.c.data(), data.height, data.width, data.k,
           bench.get_stream());
  });
  finish_matmul_summaries(state, flops);
#endif
}

inline void benchmark_matmul(
    nvbench::state& state, unsigned height, unsigned width, unsigned k, MatmulKernel launch) {
  if (height == 0 || width == 0 || k == 0) {
    state.skip("matrix dimensions must be positive");
    return;
  }

  Matmul matmul(height, width, k);
  matmul.fill();
  const auto& selected_device = state.get_device();
  if (!selected_device.has_value()) {
    state.skip("CUDA device is unavailable");
    return;
  }
  const int device = selected_device.value().get_id();
  if (!CUDA_CHECK(cudaSetDevice(device))) {
    state.skip("CUDA device selection failed");
    return;
  }
  MatmulDeviceData data(matmul, device);
  benchmark_matmul(state, data, launch);
}

inline bool
get_matmul_dimensions(nvbench::state& state, unsigned& height, unsigned& width, unsigned& k) {
  const auto axis_height = state.get_int64("Height");
  const auto axis_width = state.get_int64("Width");
  const auto axis_k = state.get_int64("K");
  constexpr auto max = static_cast<nvbench::int64_t>(std::numeric_limits<int>::max());
  if (axis_height <= 0 || axis_width <= 0 || axis_k <= 0 || axis_height > max || axis_width > max ||
      axis_k > max) {
    state.skip("matrix dimensions must fit positive unsigned values");
    return false;
  }
  height = static_cast<unsigned>(axis_height);
  width = static_cast<unsigned>(axis_width);
  k = static_cast<unsigned>(axis_k);
  return true;
}

inline int run_nvbench_args(int argc, char** argv) try { NVBENCH_MAIN_BODY(argc, argv); }
NVBENCH_MAIN_CATCH_EXCEPTIONS
#endif
