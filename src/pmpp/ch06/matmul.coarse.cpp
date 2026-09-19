#include "../ch03/matmul.hpp"

#include "matmul/benchmark.hpp"
#include "matmul/kernel.hpp"

#include <cuda/cmath>

namespace {

namespace mm = lg::matmul;

constexpr auto kTileWidth = 16U;
constexpr auto kCoarseFactor = 4U;
constexpr auto kBenchSize = 4096U;

__global__ void
matmul_coarse_kernel(const float* a, const float* b, float* c, unsigned m, unsigned n, unsigned k) {
  __shared__ float aTile[kTileWidth][kTileWidth];
  __shared__ float bTile[kTileWidth][kTileWidth];

  const auto tx = threadIdx.x;                   // Select the tile column.
  const auto ty = threadIdx.y;                   // Select the tile row.
  const auto row = blockIdx.y * kTileWidth + ty; // Select C's row.
  const auto colBase = blockIdx.x * kTileWidth * kCoarseFactor + tx;

  float sums[kCoarseFactor]{}; // Zero-initialize every accumulator.

  for (auto tileBase = 0U; tileBase < k; tileBase += kTileWidth) {
    const auto aCol = tileBase + tx; // Select A's global column.
    const auto bRow = tileBase + ty; // Select B's global row.
    aTile[ty][tx] =                  // Load or pad A.
        row < m && aCol < k ? a[row * k + aCol] : 0.0F;

    for (auto coarse = 0U; coarse < kCoarseFactor; ++coarse) {
      const auto col = colBase + coarse * kTileWidth; // Select C's column.
      bTile[ty][tx] =                                 // Load or pad B.
          bRow < k && col < n ? b[bRow * n + col] : 0.0F;
      __syncthreads(); // Publish both shared tiles.

      for (auto j = 0U; j < kTileWidth; ++j) { // Reduce one tile.
        sums[coarse] += aTile[ty][j] * bTile[j][tx];
      }
      __syncthreads(); // Protect B before replacement.
    }
  }

  if (row >= m) {
    return;
  }

  for (auto coarse = 0U; coarse < kCoarseFactor; ++coarse) {
    if (const auto col = colBase + coarse * kTileWidth; col < n) {
      c[row * n + col] = sums[coarse];
    }
  }
}

void launch_matmul_coarse(
    const float* a, const float* b, float* c, const mm::GemmShape& shape, cudaStream_t stream) {
  const dim3 block(kTileWidth, kTileWidth);
  const dim3 grid(cuda::ceil_div(shape.n(), block.x * kCoarseFactor),
                  cuda::ceil_div(shape.m(), block.y));
  matmul_coarse_kernel<<<grid, block, 0, stream>>>(a, b, c, shape.m(), shape.n(), shape.k());
}

const mm::Kernel kMatmulCoarse{"matmul.coarse", launch_matmul_coarse};

} // namespace

int main(int argc, char** argv) {
  return mm::run_app(argc, argv,
                     {kMatmulCoarse, kMatmulCell, {kBenchSize, kBenchSize, kBenchSize}});
}
