#include "../ch03/matmul.hpp"

#include "matmul/benchmark.hpp"
#include "matmul/kernel.hpp"

#include <cuda/cmath>
#include <limits>

namespace {

namespace mm = lg::matmul;

// PMPP 4e Fig. 5.9 uses 16.
constexpr unsigned kTileWidth = 16;
constexpr unsigned kBenchSize = 4096;

// PMPP 4e Fig. 5.9, §5.4–5.5.
__global__ void
matmul_tile_kernel(const float* a, const float* b, float* c, unsigned m, unsigned n, unsigned k) {
  __shared__ float aTile[kTileWidth][kTileWidth];
  __shared__ float bTile[kTileWidth][kTileWidth];
  auto gx = blockIdx.x * blockDim.x + threadIdx.x;
  auto gy = blockIdx.y * blockDim.y + threadIdx.y;
  auto tx = threadIdx.x;
  auto ty = threadIdx.y;

  float tmp = 0;
  auto tileCount = cuda::ceil_div(k, kTileWidth);
  for (auto i = 0; i < tileCount; i++) {
    // load data to SMEM, with bound check consideration.
    auto aGx = i * kTileWidth + tx;
    aTile[ty][tx] = gy < m && aGx < k ? a[gy * k + aGx] : 0;
    auto bGy = ty + i * kTileWidth;
    bTile[ty][tx] = bGy < k && gx < n ? b[n * bGy + gx] : 0;

    // wait for other threads in the tile to load their elements.
    __syncthreads();

    for (auto j = 0; j < kTileWidth; j++) {
      tmp += aTile[ty][j] * bTile[j][tx];
    }

    // wait for all threads to complete their computation
    // before starting next loop and modify the shared tiles.
    __syncthreads();
  }

  if (gy < m && gx < n) {
    c[gy * n + gx] = tmp;
  }
}

void launch_matmul_tile(
    const float* a, const float* b, float* c, const mm::GemmShape& shape, cudaStream_t stream) {
  const unsigned height = shape.m();
  const unsigned width = shape.n();
  const unsigned k = shape.k();
  const dim3 block(kTileWidth, kTileWidth);
  const dim3 grid(cuda::ceil_div(width, block.x), cuda::ceil_div(height, block.y));
  matmul_tile_kernel<<<grid, block, 0, stream>>>(a, b, c, height, width, k);
}

bool tile_traffic(const mm::GemmShape& shape, mm::Traffic& traffic) {
  const std::size_t a_reloads = cuda::ceil_div(shape.n(), kTileWidth);
  const std::size_t b_reloads = cuda::ceil_div(shape.m(), kTileWidth);
  if (shape.a_size() > std::numeric_limits<std::size_t>::max() / a_reloads ||
      shape.b_size() > std::numeric_limits<std::size_t>::max() / b_reloads) {
    return false;
  }
  const std::size_t a_reads = shape.a_size() * a_reloads;
  const std::size_t b_reads = shape.b_size() * b_reloads;
  if (a_reads > std::numeric_limits<std::size_t>::max() - b_reads) {
    return false;
  }
  traffic.global_reads = a_reads + b_reads;
  traffic.global_writes = shape.c_size();
  return true;
}

const mm::Kernel kMatmulTile{
    "matmul.tile",
    launch_matmul_tile,
    tile_traffic,
    {},
    lg::benchmark::fixed_resources<mm::GemmShape, matmul_tile_kernel, kTileWidth, kTileWidth>};

} // namespace

int main(int argc, char** argv) {
  return mm::run_app(argc, argv,
                     {kMatmulTile, kMatmulCell, mm::GemmShape(kBenchSize, kBenchSize, kBenchSize)});
}
