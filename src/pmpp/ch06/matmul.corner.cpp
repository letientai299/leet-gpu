#include "../ch03/matmul.hpp"

#include "matmul/benchmark.hpp"
#include "matmul/kernel.hpp"

#include <cuda/cmath>

namespace {

namespace mm = lg::matmul;

constexpr unsigned kTileWidth = 16;
constexpr unsigned kBenchSize = 4096;

// PMPP 4e Fig. 6.4.
// Tile/coarse load row-major B directly.
// Corner turning coalesces column-major B loads.
__global__ void
matmul_corner_kernel(const float* a, const float* b, float* c, unsigned m, unsigned n, unsigned k) {
  __shared__ float aTile[kTileWidth][kTileWidth];
  // Pad transposed writes against bank conflicts.
  __shared__ float bTile[kTileWidth][kTileWidth + 1];

  const auto gx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto gy = blockIdx.y * blockDim.y + threadIdx.y;
  const auto tx = threadIdx.x;
  const auto ty = threadIdx.y;

  float tmp = 0.0F;
  const auto tileCount = cuda::ceil_div(k, kTileWidth);
  for (auto tile = 0U; tile < tileCount; ++tile) {
    const auto aCol = tile * kTileWidth + tx;
    aTile[ty][tx] = gy < m && aCol < k ? a[gy * k + aCol] : 0.0F;

    const auto bRow = tile * kTileWidth + tx;
    const auto bCol = blockIdx.x * kTileWidth + ty;
    // Swap thread roles while loading B.
    bTile[tx][ty] = bRow < k && bCol < n ? b[bCol * k + bRow] : 0.0F;
    __syncthreads();

    for (auto j = 0U; j < kTileWidth; ++j) {
      tmp += aTile[ty][j] * bTile[j][tx];
    }
    __syncthreads();
  }

  if (gy < m && gx < n) {
    c[gy * n + gx] = tmp;
  }
}

void launch_matmul_corner(
    const float* a, const float* b, float* c, const mm::GemmShape& shape, cudaStream_t stream) {
  const dim3 block(kTileWidth, kTileWidth);
  const dim3 grid(cuda::ceil_div(shape.n(), block.x), cuda::ceil_div(shape.m(), block.y));
  matmul_corner_kernel<<<grid, block, 0, stream>>>(a, b, c, shape.m(), shape.n(), shape.k());
}

mm::Matrix column_major(const mm::Matrix& input) {
  mm::Matrix output(input.cols(), input.rows());
  for (auto row = 0U; row < input.rows(); ++row) {
    for (auto col = 0U; col < input.cols(); ++col) {
      output(col, row) = input(row, col);
    }
  }
  return output;
}

const mm::Kernel kMatmulCorner{
    "matmul.corner", launch_matmul_corner, nullptr, {nullptr, column_major}};

} // namespace

int main(int argc, char** argv) {
  return mm::run_app(argc, argv,
                     {kMatmulCorner, kMatmulCell, {kBenchSize, kBenchSize, kBenchSize}});
}
