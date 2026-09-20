#include "checks.hpp"
#include "convolution/app.hpp"

#include <cuda/cmath>

namespace {

namespace conv = lg::convolution;

__global__ void
conv_constant_kernel(const float* input, const float* filter, float* output, conv::Shape shape) {
  NOT_IMPLEMENTED();
}

void launch_constant(const float* input,
                     const float* filter,
                     float* output,
                     conv::Shape shape,
                     cudaStream_t stream) {
  constexpr dim3 block(16, 16);
  const dim3 grid(cuda::ceil_div(shape.width, block.x), cuda::ceil_div(shape.height, block.y));
  conv_constant_kernel<<<grid, block, 0, stream>>>(input, filter, output, shape);
}

} // namespace

int main(int argc, char** argv) {
  return conv::run_check(argc, argv, {"conv.constant", launch_constant});
}
