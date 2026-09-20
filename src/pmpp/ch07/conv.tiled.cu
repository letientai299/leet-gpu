#include "checks.hpp"
#include "convolution/benchmark.hpp"

#include <cuda/cmath>

namespace {

namespace conv = lg::convolution;

constexpr dim3 block_shape() {
  return {16, 16};
}

__global__ void
conv_tiled_kernel(const float* input, const float* filter, float* output, conv::Shape shape) {
  NOT_IMPLEMENTED();
}

void launch_tiled(const float* input,
                  const float* filter,
                  float* output,
                  conv::Shape shape,
                  cudaStream_t stream) {
  constexpr dim3 block = block_shape();
  const dim3 grid(cuda::ceil_div(shape.width, block.x), cuda::ceil_div(shape.height, block.y));
  conv_tiled_kernel<<<grid, block, 0, stream>>>(input, filter, output, shape);
}

} // namespace

int main(int argc, char** argv) {
  return conv::run_app(argc, argv,
                       {"conv.tiled", launch_tiled, nullptr,
                        lg::benchmark::fixed_resources<conv::Shape, conv_tiled_kernel, 16, 16>});
}
