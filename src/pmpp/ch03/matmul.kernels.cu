#include "matmul.hpp"

#include <cuda/cmath>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>

namespace {

// C[height, width] = A[height, k] * B[k, width]. One thread per output element.
__global__ void matmul_cell_kernel(
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

// Ex 3.1a: each thread produces one output matrix row.
__global__ void matmul_row_kernel(
    const float* a, const float* b, float* c, unsigned height, unsigned width, unsigned k) {
  const auto y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= height) {
    return;
  }

  for (auto x = 0U; x < width; ++x) {
    const auto idx = y * width + x;
    c[idx] = 0.0F;
    for (auto i = 0U; i < k; ++i) {
      c[idx] += a[y * k + i] * b[i * width + x];
    }
  }
}

// Ex 3.1b: each thread produces one output matrix column.
__global__ void matmul_col_kernel(
    const float* a, const float* b, float* c, unsigned height, unsigned width, unsigned k) {
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= width) {
    return;
  }

  for (auto y = 0U; y < height; ++y) {
    const auto idx = y * width + x;
    c[idx] = 0.0F;
    for (auto i = 0U; i < k; ++i) {
      c[idx] += a[y * k + i] * b[i * width + x];
    }
  }
}

} // namespace

bool Matmul::reference() {
  // CUTLASS 2.x uses SIMT FP32 and row-major layouts as the teaching kernels do.
  using RowMajor = cutlass::layout::RowMajor;
  using Gemm = cutlass::gemm::device::Gemm<float, RowMajor, float, RowMajor, float, RowMajor>;

  return on_device(expected, [&](const dbuf& dev_a, const dbuf& dev_b, dbuf& dev_c) {
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    const auto cols = static_cast<int>(width);
    const auto rows = static_cast<int>(height);
    const auto inner = static_cast<int>(k);
    const Gemm::Arguments args({rows, cols, inner}, {dev_a.data(), inner}, {dev_b.data(), cols},
                               {dev_c.data(), cols}, {dev_c.data(), cols}, {alpha, beta});
#ifdef __clang_analyzer__
    (void)args;
    return true;
#else
    Gemm gemm;
    return CUTLASS_CHECK(gemm(args));
#endif
  });
}

void launch_matmul_cell(const float* a,
                        const float* b,
                        float* c,
                        unsigned height,
                        unsigned width,
                        unsigned k,
                        cudaStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid(cuda::ceil_div(width, block.x), cuda::ceil_div(height, block.y));
  matmul_cell_kernel<<<grid, block, 0, stream>>>(a, b, c, height, width, k);
}

void launch_matmul_row(const float* a,
                       const float* b,
                       float* c,
                       unsigned height,
                       unsigned width,
                       unsigned k,
                       cudaStream_t stream) {
  constexpr unsigned block = 256;
  const auto grid = static_cast<unsigned>(cuda::ceil_div(height, block));
  matmul_row_kernel<<<grid, block, 0, stream>>>(a, b, c, height, width, k);
}

void launch_matmul_col(const float* a,
                       const float* b,
                       float* c,
                       unsigned height,
                       unsigned width,
                       unsigned k,
                       cudaStream_t stream) {
  constexpr unsigned block = 256;
  const auto grid = static_cast<unsigned>(cuda::ceil_div(width, block));
  matmul_col_kernel<<<grid, block, 0, stream>>>(a, b, c, height, width, k);
}
