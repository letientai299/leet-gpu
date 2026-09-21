#include "convolution/benchmark.hpp"
#include "convolution/constant.cuh"

#include <cuda/cmath>

namespace {

namespace conv = lg::convolution;

constexpr dim3 block_shape() {
  return {32, 32};
}

// PMPP §7.3 constant memory and caching.
//
// A100X SM80; 4096², r=3.
// Nsys: 1.260933 ms; 1302.838 GFLOP/s.
// NCU SM requests: 4.053 GB.
// NCU L2 traffic: 346.224 MB.
// NCU DRAM traffic: 124.198 MB.
__global__ void conv_constant_kernel(const float* input, float* output, conv::Shape shape) {
  // the cell I'm working on
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

  // bound check for the output cell
  if (x >= shape.width || y >= shape.height) {
    return;
  }

  float sum = 0.0f;
  for (int fy = 0; fy < shape.filter_width(); fy++) {
    for (int fx = 0; fx < shape.filter_width(); fx++) {
      const int iy = y + fy - shape.radius;
      const int ix = x + fx - shape.radius;
      if (iy >= 0 && iy < shape.height && ix >= 0 && ix < shape.width) {
        const float f = conv::constant::gFilter[fy * shape.filter_width() + fx];
        sum += input[iy * shape.width + ix] * f;
      }
    }
  }

  output[y * shape.width + x] = sum;
}

void launch_constant(
  const float* input, const float*, float* output, conv::Shape shape, cudaStream_t stream
) {
  constexpr dim3 block = block_shape();
  const dim3 grid(cuda::ceil_div(shape.width, block.x), cuda::ceil_div(shape.height, block.y));
  conv_constant_kernel<<<grid, block, 0, stream>>>(input, output, shape);
}

} // namespace

int main(int argc, char** argv) {
  return conv::run_app(
    argc, argv,
    {"conv.constant", launch_constant, conv::constant::traffic,
     lg::benchmark::fixed_resources<conv::Shape, conv_constant_kernel, 32, 32>,
     conv::constant::setup_filter}
  );
}
