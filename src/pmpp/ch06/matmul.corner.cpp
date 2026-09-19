#include "../ch03/matmul.hpp"

#include "matmul/benchmark.hpp"
#include "matmul/kernel.hpp"

#include <cuda/cmath>

namespace {

namespace mm = lg::matmul;

constexpr unsigned kTileWidth = 16;
constexpr unsigned kBenchSize = 4096;

// PMPP 4e Fig. 6.4.
// Same scalar-per-thread body as ch05 matmul.tile.
// Isolates corner turning; no thread coarsening.
// Tile kernel loads row-major B directly.
// Corner turning coalesces column-major B loads.
// Helps when B is naturally column-major (e.g. a
// transposed operand), avoiding a strided global load.
// Not needed when both operands are already row-major.
__global__ void
matmul_corner_kernel(const float* a, const float* b, float* c, unsigned m, unsigned n, unsigned k) {
  // Same aTile layout and load as matmul.tile; A is unaffected.
  __shared__ float aTile[kTileWidth][kTileWidth];
  // Pad so a warp's tx-strided writes span all 32 banks, not 2.
  // Tile kernel needs no pad: its bTile write is tx-contiguous.
  // Not from PMPP Fig. 6.4; general bank-conflict fix, see:
  // https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks
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

    // Tile kernel: bGy = ty + i*W, reads b[bGy*n + gx].
    // Here B is pre-transposed to column-major by column_major().
    const auto bRow = tile * kTileWidth + tx;
    const auto bCol = blockIdx.x * kTileWidth + ty;
    // Swap thread roles so the column-major load of B stays coalesced.
    // Read: consecutive tx -> consecutive bRow -> unit stride in memory.
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

// Host-side transpose; tile kernel takes B row-major, untransformed.
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
