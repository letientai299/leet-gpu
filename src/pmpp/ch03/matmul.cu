#include "matmul.hpp"

#include <cuda/cmath>

namespace {

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

void launch_matmul(const float* a,
                   const float* b,
                   float* c,
                   unsigned height,
                   unsigned width,
                   unsigned k,
                   cudaStream_t stream = nullptr) {
  const dim3 block(16, 16);
  const dim3 grid(cuda::ceil_div(width, block.x), cuda::ceil_div(height, block.y));
  matmul_kernel<<<grid, block, 0, stream>>>(a, b, c, height, width, k);
}

int run_matmul() {
  // Not multiples of 16: kernel must bound-check.
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
