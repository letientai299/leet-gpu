#pragma once

#include "checks.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cuda/buffer>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <random>
#include <utility>
#include <vector>

#ifdef LEET_GPU_HAS_NVBENCH
#include <nvbench/main.cuh>
#include <nvbench/nvbench.cuh>
#endif

using dbuf = cuda::device_buffer<float>;

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

inline void Matmul::fill() {
  // Fresh seed each run; values in [-1, 1].
  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  const auto next = [&] {
    return dist(rng);
  };
  std::generate(a.begin(), a.end(), next);
  std::generate(b.begin(), b.end(), next);
}

// CUTLASS 2.x device::Gemm (not 3.x CuTe / GemmUniversalAdapter). Row-major
// so no A/B swap. Default OpClassSimt is CUDA-core FFMA, not TF32.
// Links: docs/pmpp/readme.md (3.4).
using RowMajor = cutlass::layout::RowMajor;
using OracleGemm = cutlass::gemm::device::Gemm<float, RowMajor, float, RowMajor, float, RowMajor>;

inline bool Matmul::reference() {
  // Range ctor copies host A/B to device. C is write-only from the GEMM.
  return on_device(expected, [&](const dbuf& dev_a, const dbuf& dev_b, dbuf& dev_c) {
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F; // C := A*B, do not add into C
    // CUTLASS wants int dims. Row-major leading dims: k for A, width for B
    // and C. C and D alias; beta 0 means C is never read.
    const auto cols = static_cast<int>(width);
    const auto rows = static_cast<int>(height);
    const auto inner = static_cast<int>(k);
    const OracleGemm::Arguments args({rows, cols, inner}, {dev_a.data(), inner},
                                     {dev_b.data(), cols}, {dev_c.data(), cols},
                                     {dev_c.data(), cols}, {alpha, beta});
#ifdef __clang_analyzer__
    (void)args;
    return true;
#else
    OracleGemm gemm;
    return CUTLASS_CHECK(gemm(args));
#endif
  });
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
template <typename Launch>
void benchmark_matmul(
    nvbench::state& state, unsigned height, unsigned width, unsigned k, Launch&& launch) {
  Matmul matmul(height, width, k);
  matmul.fill();

  const auto stream = default_stream();
  auto& pool = device_pool();
  const dbuf a{stream, pool, matmul.a};
  const dbuf b{stream, pool, matmul.b};
  dbuf c{stream, pool, matmul.c};
  float* const output = c.data();
  if (!CUDA_CHECK(cudaDeviceSynchronize())) {
    state.skip("CUDA setup failed");
    return;
  }

  const auto outputs = static_cast<std::size_t>(height) * width;
  const auto reads = outputs * k * 2;
  state.add_element_count(outputs);
  state.add_global_memory_reads<float>(reads);
  state.add_global_memory_writes<float>(outputs);
#ifdef __clang_analyzer__
  output[0] = 0.0F;
  launch(a.data(), b.data(), output, height, width, k, nullptr);
#else
  state.exec(nvbench::exec_tag::gpu, [&](nvbench::launch& bench) {
    launch(a.data(), b.data(), output, height, width, k, bench.get_stream());
  });
#endif
}

inline int run_nvbench(int argc, char** argv) try {
  std::vector<char*> args(argv, argv + argc);
  args.erase(args.begin() + 1);
  int const bench_argc = static_cast<int>(args.size());
  char** const bench_argv = args.data();
  NVBENCH_MAIN_BODY(bench_argc, bench_argv);
}
NVBENCH_MAIN_CATCH_EXCEPTIONS

template <typename Fn> int run_matmul_app(int argc, char** argv, Fn&& body) {
  if (help_requested(argc, argv)) {
    std::printf("Usage: %s [--bench [options]]\n", argv[0]);
    return 0;
  }
  const bool bench = argc >= 2 && std::strcmp(argv[1], "--bench") == 0;
  if (argc != 1 && !bench) {
    std::printf("Usage: %s [--bench [options]]\n", argv[0]);
    return 2;
  }

  const int result = run_host(1, argv, std::forward<Fn>(body));
  return result != 0 || !bench ? result : run_nvbench(argc, argv);
}
#endif
