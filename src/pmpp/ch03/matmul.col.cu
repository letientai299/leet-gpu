#include "matmul.hpp"

#include <cuda/cmath>

namespace {

// Ex 3.1b: each thread produces one output matrix column.
__global__ void matmul_kernel(
    const float* a, const float* b, float* c, unsigned height, unsigned width, unsigned k) {
}

int run_matmul() {
  // Not a multiple of 256: kernel must bound-check.
  constexpr unsigned height = 67;
  constexpr unsigned width = 33;
  constexpr unsigned k = 50;
  return Matmul(height, width, k)
      .run([](const dbuf& a, const dbuf& b, dbuf& c, unsigned height, unsigned width, unsigned k) {
        constexpr unsigned block = 256;
        const auto grid = static_cast<unsigned>(cuda::ceil_div(width, block));
        matmul_kernel<<<grid, block>>>(a.data(), b.data(), c.data(), height, width, k);
        return CUDA_CHECK(cudaGetLastError());
      });
}

} // namespace

int main(int argc, char** argv) {
  return run_host(argc, argv, run_matmul);
}
