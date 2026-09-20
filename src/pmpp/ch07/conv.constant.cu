#include "convolution/benchmark.hpp"

#include <cuda/cmath>

namespace {

namespace conv = lg::convolution;

constexpr int kMaxFilterRadius = 7;
constexpr int kMaxFilterWidth = kMaxFilterRadius * 2 + 1;
__constant__ float gFilter[kMaxFilterWidth * kMaxFilterWidth]{};

constexpr dim3 block_shape() {
  return {32, 32};
}

bool constant_traffic(conv::Shape shape, conv::Traffic& traffic) {
  const auto reads = conv::checked_mul(
    conv::valid_axis_pairs(shape.width, shape.radius),
    conv::valid_axis_pairs(shape.height, shape.radius)
  );
  if (!reads) {
    return false;
  }
  traffic.global_reads = *reads;
  traffic.global_writes = shape.input_size();
  return true;
}

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
        const float f = gFilter[fy * shape.filter_width() + fx];
        sum += input[iy * shape.width + ix] * f;
      }
    }
  }

  output[y * shape.width + x] = sum;
}

cudaError_t setup_filter(const float* filter, conv::Shape shape, cudaStream_t stream) {
  if (shape.radius < 0 || shape.radius > kMaxFilterRadius) {
    return cudaErrorInvalidValue;
  }
  return cudaMemcpyToSymbolAsync(
    gFilter, filter, shape.filter_size() * sizeof(float), 0, cudaMemcpyDeviceToDevice, stream
  );
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
    {"conv.constant", launch_constant, constant_traffic,
     lg::benchmark::fixed_resources<conv::Shape, conv_constant_kernel, 32, 32>, setup_filter}
  );
}
