#include "stencil/benchmark.hpp"

#include <cstddef>
#include <cuda/cmath>

namespace {

namespace st = lg::stencil;

constexpr dim3 block_shape() {
  return {8, 8, 8};
}

// PMPP §8.2 basic seven-point stencil (Fig. 8.6).
// Write interior points only; the harness keeps boundary values.
__global__ void
stencil_basic(const float* input, float* output, st::Shape shape, st::Coefficients coeff) {
  const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= 1 && x < shape.width - 1 &&  //
      y >= 1 && y < shape.height - 1 && //
      z >= 1 && z < shape.depth - 1) {
    const unsigned out = z * shape.plane_size() + static_cast<std::size_t>(y * shape.width) + x;
    // Float ops: 7 mul, 6 add
    // Bytes read: 7*4
    // Op/B = 13/28 = 0.46
    output[out] = coeff.c0 * input[out] +                      //
                  coeff.c1 * input[out - 1] +                  //
                  coeff.c2 * input[out + 1] +                  //
                  coeff.c3 * input[out - shape.width] +        //
                  coeff.c4 * input[out + shape.width] +        //
                  coeff.c5 * input[out - shape.plane_size()] + //
                  coeff.c6 * input[out + shape.plane_size()];
  }
}

void launch_basic(
  const float* input,
  float* output,
  st::Shape shape,
  st::Coefficients coefficients,
  cudaStream_t stream
) {
  constexpr dim3 block = block_shape();
  const dim3 grid(
    cuda::ceil_div(shape.width, block.x),  //
    cuda::ceil_div(shape.height, block.y), //
    cuda::ceil_div(shape.depth, block.z)
  );
  stencil_basic<<<grid, block, 0, stream>>>(input, output, shape, coefficients);
}

} // namespace

int main(int argc, char** argv) {
  return st::run_app(
    argc, argv,
    {.name = "stencil.basic",
     .launch = launch_basic,
     .traffic = st::basic_traffic,
     .resources = lg::benchmark::fixed_resources<st::Shape, stencil_basic, block_shape()>}
  );
}
