#include "convolution/app.hpp"

#include <cuda/cmath>

namespace {

namespace conv = lg::convolution;

__global__ void conv_basic_kernel(const float* input, //
                                  const float* filter,
                                  float* output,
                                  conv::Shape shape) {
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

void launch_basic(const float* input,
                  const float* filter,
                  float* output,
                  conv::Shape shape,
                  cudaStream_t stream) {
  constexpr dim3 block(32, 32);
  const dim3 grid(cuda::ceil_div(shape.width, block.x), //
                  cuda::ceil_div(shape.height, block.y));
  conv_basic_kernel<<<grid, block, 0, stream>>>(input, filter, output, shape);
}

} // namespace

int main(int argc, char** argv) {
  return conv::run_check(argc, argv, {"conv.basic", launch_basic});
}
