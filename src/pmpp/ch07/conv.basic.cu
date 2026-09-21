#include "convolution/benchmark.hpp"

#include <cuda/cmath>

namespace {

namespace conv = lg::convolution;

constexpr dim3 block_shape() {
  return {32, 32};
}

// PMPP §7.2 parallel convolution: a basic algorithm.
//
// A100X SM80; 4096², r=3.
// Nsys: 1.297349 ms; 1266.268 GFLOP/s.
// NCU SM requests: 4.875 GB.
// NCU L2 traffic: 344.885 MB.
// NCU DRAM traffic: 124.200 MB.
__global__ void
conv_basic_kernel(const float* input, const float* filter, float* output, conv::Shape shape) {
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
        auto f = filter[fy * shape.filter_width() + fx];
        sum += input[iy * shape.width + ix] * f;
      }
    }
  }

  output[y * shape.width + x] = sum;
}

void launch_basic(
  const float* input, const float* filter, float* output, conv::Shape shape, cudaStream_t stream
) {
  constexpr dim3 block = block_shape();
  const dim3 grid(cuda::ceil_div(shape.width, block.x), cuda::ceil_div(shape.height, block.y));
  conv_basic_kernel<<<grid, block, 0, stream>>>(input, filter, output, shape);
}

} // namespace

int main(int argc, char** argv) {
  return conv::run_app(
    argc, argv,
    {"conv.basic", launch_basic, conv::basic_traffic,
     lg::benchmark::fixed_resources<conv::Shape, conv_basic_kernel, 32, 32>}
  );
}
