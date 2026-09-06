#include "matmul.hpp"

#include <cuda/cmath>

namespace {

// Ex 3.1b: each thread produces one output matrix column.
__global__ void matmul_col_kernel(
    const float* a, const float* b, float* c, unsigned height, unsigned width, unsigned k) {
  // launch is 1d block covering the x-index of C
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= width) {
    return;
  }

  for (auto y = 0; y < height; y++) {
    const auto idx = y * width + x;
    c[idx] = 0; // defensive

    // compute the cell C[x, y]
    for (auto i = 0; i < k; i++) {
      c[idx] += a[y * k + i] * b[i * width + x];
    }
  }
}

void launch_matmul(const float* a,
                   const float* b,
                   float* c,
                   unsigned height,
                   unsigned width,
                   unsigned k,
                   cudaStream_t stream = nullptr) {
  constexpr unsigned block = 256;
  const auto grid = static_cast<unsigned>(cuda::ceil_div(width, block));
  matmul_col_kernel<<<grid, block, 0, stream>>>(a, b, c, height, width, k);
}

int run_matmul() {
  // Not a multiple of 256: kernel must bound-check.
  constexpr unsigned height = 67;
  constexpr unsigned width = 33;
  constexpr unsigned k = 50;
  return Matmul(height, width, k)
      .run([](const dbuf& a, const dbuf& b, dbuf& c, unsigned height, unsigned width, unsigned k) {
        launch_matmul(a.data(), b.data(), c.data(), height, width, k);
        return CUDA_CHECK(cudaGetLastError());
      });
}

void bench_matmul(nvbench::state& state) {
  benchmark_matmul(state, 67, 33, 50, launch_matmul);
}

NVBENCH_BENCH(bench_matmul).set_name("matmul");

} // namespace

int main(int argc, char** argv) {
  return run_matmul_app(argc, argv, run_matmul);
}
