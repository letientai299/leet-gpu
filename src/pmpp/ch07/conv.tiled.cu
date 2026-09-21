#include "convolution/benchmark.hpp"
#include "convolution/constant.cuh"

#include <cuda/cmath>

namespace {

namespace conv = lg::convolution;

constexpr int kTile = 16;
constexpr int kMaxInTile = kTile + 2 * conv::constant::kMaxFilterRadius;

constexpr dim3 block_shape() {
  return {kTile, kTile};
}

// PMPP §7.4 tiled convolution with halo cells.
//
// A100X SM80; 4096², r=3.
// Nsys: 0.854467 ms; 1924.202 GFLOP/s.
// NCU SM requests: 297.140 MB.
// NCU L2 traffic: 433.501 MB.
// NCU DRAM traffic: 124.723 MB.
__global__ void conv_tiled_kernel(const float* input, float* output, conv::Shape shape) {
  __shared__ float tile[kMaxInTile][kMaxInTile];

  // the cell
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

  // tile indice
  const int ty = static_cast<int>(threadIdx.y);
  const int tx = static_cast<int>(threadIdx.x);

  const int in_tile = kTile + 2 * shape.radius;
  const int y0 = y - ty;
  const int x0 = x - tx;
  for (int i = ty; i < in_tile; i += kTile) {
    for (int j = tx; j < in_tile; j += kTile) {
      const int iy = y0 + i - shape.radius;
      const int ix = x0 + j - shape.radius;
      tile[i][j] = (iy >= 0 && iy < shape.height && ix >= 0 && ix < shape.width)
                     ? input[iy * shape.width + ix]
                     : 0.0f;
    }
  }
  __syncthreads();

  // bound check for the output cell
  if (x >= shape.width || y >= shape.height) {
    return;
  }

  float sum = 0.0f;
  for (int fy = 0; fy < shape.filter_width(); fy++) {
    for (int fx = 0; fx < shape.filter_width(); fx++) {
      const float f = conv::constant::gFilter[fy * shape.filter_width() + fx];
      // halo is already in this tile (0 if ghost), so every tap is SMEM.
      sum += tile[ty + fy][tx + fx] * f;
    }
  }

  output[y * shape.width + x] = sum;
}

void launch_tiled(
  const float* input, const float*, float* output, conv::Shape shape, cudaStream_t stream
) {
  constexpr dim3 block = block_shape();
  const dim3 grid(cuda::ceil_div(shape.width, block.x), cuda::ceil_div(shape.height, block.y));
  conv_tiled_kernel<<<grid, block, 0, stream>>>(input, output, shape);
}

} // namespace

int main(int argc, char** argv) {
  return conv::run_app(
    argc, argv,
    // Reports uncached tap traffic.
    {"conv.tiled", launch_tiled, conv::constant::traffic,
     lg::benchmark::fixed_resources<conv::Shape, conv_tiled_kernel, 16, 16>,
     conv::constant::setup_filter}
  );
}
