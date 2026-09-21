#include "convolution3d/benchmark.hpp"

#include <cuda/cmath>

namespace {

namespace conv = lg::convolution3d;

constexpr dim3 block_shape() {
  return {8, 8, 8};
}

// PMPP Ex 7.8 3D revision of the §7.2 basic kernel.
//
// A100X SM80 at 1.215 GHz; 256×256×128, r=3.
// Cold: 3.462 ms; 1617.8 GFLOP/s.
// Batch: 3.459 ms; 1619.4 GFLOP/s.
// NVBench converged: 0.02% cold GPU noise.
__global__ void
conv3d_basic_kernel(const float* input, const float* filter, float* output, conv::Shape shape) {
  // the cell I'm working on
  const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int fw = shape.filter_width();

  // bound check for the output cell
  if (x >= shape.width || y >= shape.height || z >= shape.depth) {
    return;
  }

  float sum = 0.0f;
  for (int fz = 0; fz < fw; fz++) {
    for (int fy = 0; fy < fw; fy++) {
      for (int fx = 0; fx < fw; fx++) {
        const int iz = z + fz - shape.radius;
        const int iy = y + fy - shape.radius;
        const int ix = x + fx - shape.radius;

        if (iz >= 0 && iz < shape.depth &&  //
            iy >= 0 && iy < shape.height && //
            ix >= 0 && ix < shape.width) {
          auto f = filter[fz * fw * fw + fy * fw + fx];
          sum += input[iz * shape.width * shape.height + iy * shape.width + ix] * f;
        }
      }
    }
  }

  output[z * shape.width * shape.height + y * shape.width + x] = sum;
}

void launch_basic(
  const float* input, const float* filter, float* output, conv::Shape shape, cudaStream_t stream
) {
  constexpr dim3 block = block_shape();
  const dim3 grid(
    cuda::ceil_div(shape.width, block.x),  //
    cuda::ceil_div(shape.height, block.y), //
    cuda::ceil_div(shape.depth, block.z)
  );
  conv3d_basic_kernel<<<grid, block, 0, stream>>>(input, filter, output, shape);
}

} // namespace

int main(int argc, char** argv) {
  return conv::run_app(
    argc, argv,
    {"conv3d.basic", launch_basic, conv::basic_traffic,
     lg::benchmark::fixed_resources<conv::Shape, conv3d_basic_kernel, 8, 8, 8>}
  );
}
