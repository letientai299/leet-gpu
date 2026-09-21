#include "convolution3d/benchmark.hpp"

#include <cuda/cmath>

namespace {

namespace conv = lg::convolution3d;

constexpr int kMaxFilterRadius = 7;
constexpr int kMaxFilterWidth = kMaxFilterRadius * 2 + 1;

__constant__ float gFilter[kMaxFilterWidth * kMaxFilterWidth * kMaxFilterWidth]{};

constexpr dim3 block_shape() {
  return {8, 8, 8};
}

bool constant_traffic(conv::Shape shape, conv::Traffic& traffic) {
  const auto xy = conv::checked_mul(
    conv::valid_axis_pairs(shape.width, shape.radius),
    conv::valid_axis_pairs(shape.height, shape.radius)
  );
  const auto reads =
    xy ? conv::checked_mul(*xy, conv::valid_axis_pairs(shape.depth, shape.radius)) : std::nullopt;
  if (!reads) {
    return false;
  }
  traffic.global_reads = *reads;
  traffic.global_writes = shape.input_size();
  return true;
}

cudaError_t setup_filter(const float* filter, conv::Shape shape, cudaStream_t stream) {
  if (shape.radius < 0 || shape.radius > kMaxFilterRadius) {
    return cudaErrorInvalidValue;
  }
  return cudaMemcpyToSymbolAsync(
    gFilter, filter, shape.filter_size() * sizeof(float), 0, cudaMemcpyDeviceToDevice, stream
  );
}

// PMPP Ex 7.9 3D revision of the §7.3 constant kernel.
//
// A100X SM80 at 1.215 GHz; 256×256×128, r=3.
// Cold: 2.809 ms; 1994.2 GFLOP/s.
// Batch: 2.801 ms; 1999.7 GFLOP/s.
// NVBench converged: 0.27% cold GPU noise.
__global__ void conv3d_constant_kernel(const float* input, float* output, conv::Shape shape) {
  const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int fw = shape.filter_width();

  if (x >= shape.width || y >= shape.height || z >= shape.depth) {
    return;
  }

  float sum = 0.0F;
  for (int fz = 0; fz < fw; ++fz) {
    for (int fy = 0; fy < fw; ++fy) {
      for (int fx = 0; fx < fw; ++fx) {
        const int iz = z + fz - shape.radius;
        const int iy = y + fy - shape.radius;
        const int ix = x + fx - shape.radius;
        if (iz >= 0 && iz < shape.depth && iy >= 0 && iy < shape.height && ix >= 0 &&
            ix < shape.width) {
          const float filter_value = gFilter[(fz * fw + fy) * fw + fx];
          const int input_index = (iz * shape.height + iy) * shape.width + ix;
          sum += input[input_index] * filter_value;
        }
      }
    }
  }

  output[(z * shape.height + y) * shape.width + x] = sum;
}

void launch_constant(
  const float* input, const float*, float* output, conv::Shape shape, cudaStream_t stream
) {
  constexpr dim3 block = block_shape();
  const dim3 grid(
    cuda::ceil_div(shape.width, block.x), cuda::ceil_div(shape.height, block.y),
    cuda::ceil_div(shape.depth, block.z)
  );
  conv3d_constant_kernel<<<grid, block, 0, stream>>>(input, output, shape);
}

} // namespace

int main(int argc, char** argv) {
  return conv::run_app(
    argc, argv,
    {"conv3d.constant", launch_constant, constant_traffic,
     lg::benchmark::fixed_resources<conv::Shape, conv3d_constant_kernel, 8, 8, 8>, setup_filter}
  );
}
