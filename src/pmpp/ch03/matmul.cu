#include "checks.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cuda/buffer>
#include <cuda/cmath>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <random>
#include <vector>

namespace {

using dbuf = cuda::device_buffer<float>;

// C[height, width] = A[height, k] * B[k, width]. One thread per output element.
__global__ void matmul_kernel(
    const float* a, const float* b, float* c, unsigned height, unsigned width, unsigned k) {
  const auto col = blockIdx.x * blockDim.x + threadIdx.x;
  const auto row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col >= width || row >= height) {
    return;
  }

  const auto idx = row * width + col;
  auto sum = 0.0F;
  for (auto i = 0; i < k; ++i) {
    sum += a[row * k + i] * b[width * i + col];
  }
  c[idx] = sum;
}

// Host driver: random A/B, CUTLASS oracle, then the kernel under test.
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
  /// Launch the naive kernel and write C into c.
  bool multiply_on_gpu();
  /// True if c matches expected within atol/rtol.
  [[nodiscard]] bool verify() const;

  /// Copy A/B to the device, run op on C, copy C into out.
  template <typename Op> bool on_device(std::vector<float>& out, Op&& op) {
    const auto stream = default_stream();
    auto& pool = device_pool();
    const dbuf dev_a{stream, pool, a};
    const dbuf dev_b{stream, pool, b};
    dbuf dev_c{stream, pool, out.size(), cuda::no_init};
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

  /// Run oracle, kernel, and verify. Returns 0 on success.
  int run() {
    return init() && multiply_on_gpu() && verify() ? 0 : 1;
  }
};

void Matmul::fill() {
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

bool Matmul::reference() {
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

bool Matmul::multiply_on_gpu() {
  return on_device(c, [&](const dbuf& dev_a, const dbuf& dev_b, dbuf& dev_c) {
    const dim3 block(16, 16);
    const dim3 grid(cuda::ceil_div(width, block.x), cuda::ceil_div(height, block.y));
    matmul_kernel<<<grid, block>>>(dev_a.data(), dev_b.data(), dev_c.data(), height, width, k);
    return CUDA_CHECK(cudaGetLastError());
  });
}

bool Matmul::verify() const {
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

int run_matmul() {
  // Not multiples of 16: kernel must bound-check.
  constexpr unsigned height = 67;
  constexpr unsigned width = 33;
  constexpr unsigned k = 50;
  return Matmul(height, width, k).run();
}

} // namespace

int main(int argc, char** argv) {
  return run_host(argc, argv, run_matmul);
}
