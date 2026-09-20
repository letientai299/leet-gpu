#include "matmul.hpp"

#include <cuda/cmath>

namespace {

// C[height, width] = A[height, k] * B[k, width]. One thread per output element.
__global__ void matmul_cell_kernel(
  const float* a, const float* b, float* c, unsigned height, unsigned width, unsigned k
) {
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
  const float* a, const float* b, float* c, unsigned height, unsigned width, unsigned k
) {
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
  const float* a, const float* b, float* c, unsigned height, unsigned width, unsigned k
) {
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

void launch_matmul_cell(
  const float* a, const float* b, float* c, const lg::matmul::GemmShape& shape, cudaStream_t stream
) {
  const unsigned height = shape.m();
  const unsigned width = shape.n();
  const unsigned k = shape.k();
  const dim3 block(16, 16);
  const dim3 grid(cuda::ceil_div(width, block.x), cuda::ceil_div(height, block.y));
  matmul_cell_kernel<<<grid, block, 0, stream>>>(a, b, c, height, width, k);
}

void launch_matmul_row(
  const float* a, const float* b, float* c, const lg::matmul::GemmShape& shape, cudaStream_t stream
) {
  const unsigned height = shape.m();
  const unsigned width = shape.n();
  const unsigned k = shape.k();
  constexpr unsigned block = 256;
  const auto grid = static_cast<unsigned>(cuda::ceil_div(height, block));
  matmul_row_kernel<<<grid, block, 0, stream>>>(a, b, c, height, width, k);
}

void launch_matmul_col(
  const float* a, const float* b, float* c, const lg::matmul::GemmShape& shape, cudaStream_t stream
) {
  const unsigned height = shape.m();
  const unsigned width = shape.n();
  const unsigned k = shape.k();
  constexpr unsigned block = 256;
  const auto grid = static_cast<unsigned>(cuda::ceil_div(width, block));
  matmul_col_kernel<<<grid, block, 0, stream>>>(a, b, c, height, width, k);
}

cudaError_t matmul_cell_resources(
  const lg::matmul::GemmShape& shape, lg::benchmark::KernelResources& resources
) {
  return lg::benchmark::fixed_resources<lg::matmul::GemmShape, matmul_cell_kernel, 16, 16>(
    shape, resources
  );
}

cudaError_t matmul_row_resources(
  const lg::matmul::GemmShape& shape, lg::benchmark::KernelResources& resources
) {
  return lg::benchmark::fixed_resources<lg::matmul::GemmShape, matmul_row_kernel, 256>(
    shape, resources
  );
}

cudaError_t matmul_col_resources(
  const lg::matmul::GemmShape& shape, lg::benchmark::KernelResources& resources
) {
  return lg::benchmark::fixed_resources<lg::matmul::GemmShape, matmul_col_kernel, 256>(
    shape, resources
  );
}
