#include "matmul.hpp"

#include <cuda/cmath>

namespace {

// Ex 3.1a: each thread produces one output matrix row.
__global__ void matmul_row_kernel(
    const float* a, const float* b, float* c, unsigned height, unsigned width, unsigned k) {
  // launch is 1d block covering the y-index of C
  const auto y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= height) {
    return;
  }

  for (auto x = 0; x < width; x++) {
    const auto idx = y * width + x;
    c[idx] = 0; // defensive

    // compute the cell C[x, y]
    for (auto i = 0; i < k; i++) {
      c[idx] += a[y * k + i] * b[i * width + x];
    }
  }
}

int run_matmul() {
  // Not a multiple of 256: kernel must bound-check.
  constexpr unsigned height = 67;
  constexpr unsigned width = 33;
  constexpr unsigned k = 50;
  return Matmul(height, width, k)
      .run([](const dbuf& a, const dbuf& b, dbuf& c, unsigned height, unsigned width, unsigned k) {
        constexpr unsigned block = 256;
        const auto grid = static_cast<unsigned>(cuda::ceil_div(height, block));
        matmul_row_kernel<<<grid, block>>>(a.data(), b.data(), c.data(), height, width, k);
        return CUDA_CHECK(cudaGetLastError());
      });
}

} // namespace

int main(int argc, char** argv) {
  return run_host(argc, argv, run_matmul);
}
