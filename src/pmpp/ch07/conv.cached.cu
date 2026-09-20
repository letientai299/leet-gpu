#include "checks.hpp"
#include "convolution/benchmark.hpp"

#include <cuda/cmath>

namespace {

namespace conv = lg::convolution;

__global__ void
conv_cached_kernel(const float* input, const float* filter, float* output, conv::Shape shape) {
  NOT_IMPLEMENTED();
}

void launch_cached(const float* input,
                   const float* filter,
                   float* output,
                   conv::Shape shape,
                   cudaStream_t stream) {
  constexpr dim3 block(16, 16);
  const dim3 grid(cuda::ceil_div(shape.width, block.x), cuda::ceil_div(shape.height, block.y));
  conv_cached_kernel<<<grid, block, 0, stream>>>(input, filter, output, shape);
}

} // namespace

int main(int argc, char** argv) {
  return conv::run_app(argc, argv, {"conv.cached", launch_cached});
}
