#include "convolution/benchmark.hpp"
#include "convolution/constant.cuh"

#include <cuda/cmath>

namespace {

namespace conv = lg::convolution;

constexpr int kTile = 16;
constexpr int kMaxInTile = kTile + 2 * conv::constant::kMaxFilterRadius;

// PMPP §7.4 Fig. 7.12: thread block matches the input tile.
//
// A100X SM80; 4096 × 4096, r=3; GPU locked to 1215 MHz.
// NVBench median: 0.718848 ms; 2286.495 GFLOP/s.
// NCU SM requests: 290.417 MB.
// NCU L2 traffic: 440.774 MB.
// NCU DRAM traffic: 124.758 MB.
__global__ void conv_tiled_in_kernel(const float* input, float* output, conv::Shape shape) {
  __shared__ float tile[kMaxInTile][kMaxInTile];

  const int ty = static_cast<int>(threadIdx.y);
  const int tx = static_cast<int>(threadIdx.x);
  const int y0 = static_cast<int>(blockIdx.y) * kTile;
  const int x0 = static_cast<int>(blockIdx.x) * kTile;
  const int iy = y0 + ty - shape.radius;
  const int ix = x0 + tx - shape.radius;

  tile[ty][tx] = (iy >= 0 && iy < shape.height && ix >= 0 && ix < shape.width)
                   ? input[iy * shape.width + ix]
                   : 0.0f;
  __syncthreads();

  if (ty < shape.radius || ty >= kTile + shape.radius || tx < shape.radius ||
      tx >= kTile + shape.radius) {
    return;
  }

  const int y = y0 + ty - shape.radius;
  const int x = x0 + tx - shape.radius;
  if (x >= shape.width || y >= shape.height) {
    return;
  }

  float sum = 0.0f;
  for (int fy = 0; fy < shape.filter_width(); fy++) {
    for (int fx = 0; fx < shape.filter_width(); fx++) {
      const float f = conv::constant::gFilter[fy * shape.filter_width() + fx];
      sum += tile[ty + fy - shape.radius][tx + fx - shape.radius] * f;
    }
  }

  output[y * shape.width + x] = sum;
}

void launch_tiled_in(
  const float* input, const float*, float* output, conv::Shape shape, cudaStream_t stream
) {
  const auto in_tile = static_cast<unsigned>(kTile + 2 * shape.radius);
  const dim3 block(in_tile, in_tile);
  const dim3 grid(
    cuda::ceil_div(shape.width, static_cast<unsigned>(kTile)),
    cuda::ceil_div(shape.height, static_cast<unsigned>(kTile))
  );
  conv_tiled_in_kernel<<<grid, block, 0, stream>>>(input, output, shape);
}

cudaError_t
tiled_in_resources(const conv::Shape& shape, lg::benchmark::KernelResources& resources) {
  const auto in_tile = static_cast<unsigned>(kTile + 2 * shape.radius);
  return lg::benchmark::inspect_kernel<conv_tiled_in_kernel>(dim3{in_tile, in_tile}, 0, resources);
}

} // namespace

int main(int argc, char** argv) {
  return conv::run_app(
    argc, argv,
    {"conv.tiled.in", launch_tiled_in, conv::constant::traffic, tiled_in_resources,
     conv::constant::setup_filter}
  );
}
