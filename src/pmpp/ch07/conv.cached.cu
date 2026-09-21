#include "convolution/benchmark.hpp"
#include "convolution/constant.cuh"

#include <cuda/cmath>

namespace {

namespace conv = lg::convolution;

constexpr int kTile = 16;

constexpr dim3 block_shape() {
  return {kTile, kTile};
}

// PMPP §7.5 tiled convolution using caches for halo cells.
//
// A100X SM80; 4096 × 4096, r=3; GPU locked to 1215 MHz.
// NVBench median: 1.337344 ms; 1229.058 GFLOP/s.
// NCU SM requests: 1.887 GB.
// NCU L2 traffic: 426.582 MB.
// NCU DRAM traffic: 124.784 MB.
// L2 halo hits are not guaranteed.
__global__ void conv_cached_kernel(const float* input, float* output, conv::Shape shape) {
  __shared__ float tile[kTile][kTile];
  // the cell
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

  // tile indice
  const int ty = static_cast<int>(threadIdx.y);
  const int tx = static_cast<int>(threadIdx.x);

  tile[ty][tx] = (y < shape.height && x < shape.width) ? input[y * shape.width + x] : 0.0f;
  __syncthreads();

  // bound check for the output cell
  if (x >= shape.width || y >= shape.height) {
    return;
  }

  float sum = 0.0f;
  for (int fy = 0; fy < shape.filter_width(); fy++) {
    for (int fx = 0; fx < shape.filter_width(); fx++) {
      const int tile_y = ty + fy - shape.radius;
      const int tile_x = tx + fx - shape.radius;
      const float f = conv::constant::gFilter[fy * shape.filter_width() + fx];

      // tap sits in this block's output tile: SMEM already has it (0 if the
      // cell was outside the input), so skip the GMEM path and bound check.
      if (tile_y >= 0 && tile_y < kTile && tile_x >= 0 && tile_x < kTile) {
        sum += tile[tile_y][tile_x] * f;
      } else {
        // Halo cells can reuse L2 cache.
        const int iy = y + fy - shape.radius;
        const int ix = x + fx - shape.radius;
        if (iy >= 0 && iy < shape.height && ix >= 0 && ix < shape.width) {
          sum += input[iy * shape.width + ix] * f;
        }
      }
    }
  }

  output[y * shape.width + x] = sum;
}

void launch_cached(
  const float* input, const float*, float* output, conv::Shape shape, cudaStream_t stream
) {
  constexpr dim3 block = block_shape();
  const dim3 grid(cuda::ceil_div(shape.width, block.x), cuda::ceil_div(shape.height, block.y));
  conv_cached_kernel<<<grid, block, 0, stream>>>(input, output, shape);
}

} // namespace

int main(int argc, char** argv) {
  return conv::run_app(
    argc, argv,
    // Reports uncached tap traffic.
    {"conv.cached", launch_cached, conv::constant::traffic,
     lg::benchmark::fixed_resources<conv::Shape, conv_cached_kernel, 16, 16>,
     conv::constant::setup_filter}
  );
}
