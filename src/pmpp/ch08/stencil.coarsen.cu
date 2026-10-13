// Harness supplies shapes, coefficients, and validation.
#include "stencil/benchmark.hpp"

#include <cuda/cmath>

namespace {

namespace st = lg::stencil;

// Two halo cells leave 30 interior cells.
constexpr unsigned kInputTile = 32;
constexpr unsigned kOutputTile = kInputTile - 2;

constexpr dim3 block_shape() {
  // Threads cover XY; each loops over Z.
  return {kInputTile, kInputTile, 1};
}

// PMPP §8.4 thread coarsening (Fig. 8.10).
__global__ void stencil_coarsen(const float* in, float* out, st::Shape sh, st::Coefficients co) {
  // Advance by interiors so neighboring halos overlap.
  const unsigned x = blockIdx.x * kOutputTile + threadIdx.x;
  const unsigned y = blockIdx.y * kOutputTile + threadIdx.y;
  const unsigned zStart = blockIdx.z * kOutputTile;
  const unsigned tx = threadIdx.x;
  const unsigned ty = threadIdx.y;
  const bool validXY = x < sh.width && y < sh.height;

  // Halo threads load neighbors without writing outputs.
  const bool outputXY = tx > 0 && tx < kInputTile - 1 && ty > 0 && ty < kInputTile - 1 &&
                        x < sh.width - 1 && y < sh.height - 1;

  // Reuse three planes instead of a cube.
  __shared__ float inPrev[kInputTile][kInputTile];
  __shared__ float inCur[kInputTile][kInputTile];
  __shared__ float inNext[kInputTile][kInputTile];

  const auto plane = static_cast<unsigned>(sh.plane_size());
  // Start one plane above the lower halo.
  auto idx = (zStart + 1) * plane + y * sh.width + x;

  // Initialize skipped cells to avoid undefined reads.
  inPrev[ty][tx] = validXY ? in[idx - plane] : 0.0f;
  inCur[ty][tx] = validXY ? in[idx] : 0.0f;

  // Uniform Z bounds keep barriers safe.
  for (unsigned currentZ = zStart + 1;                 // Skip the lower halo plane.
       currentZ < zStart + 1 + kOutputTile &&          // Stop before the next block's outputs.
       currentZ < static_cast<unsigned>(sh.depth - 1); // Exclude the grid's upper boundary.
       ++currentZ, idx += plane) {                     // Advance Z and its flattened index.
    // Only the next plane needs fresh input.
    inNext[ty][tx] = validXY ? in[idx + plane] : 0.0f;
    // Publish loaded and rotated planes before computation.
    __syncthreads();

    if (outputXY) {
      out[idx] = co.c0 * inCur[ty][tx] +     // Center.
                 co.c1 * inCur[ty][tx - 1] + // X minus.
                 co.c2 * inCur[ty][tx + 1] + // X plus.
                 co.c3 * inCur[ty - 1][tx] + // Y minus.
                 co.c4 * inCur[ty + 1][tx] + // Y plus.
                 co.c5 * inPrev[ty][tx] +    // Z minus.
                 co.c6 * inNext[ty][tx];     // Z plus.
    }

    // Finish neighbor reads before overwriting shared planes.
    __syncthreads();
    // Next iteration's barrier publishes rotated planes.
    inPrev[ty][tx] = inCur[ty][tx];
    inCur[ty][tx] = inNext[ty][tx];
  }
}

void launch_coarsen(
  const float* input,
  float* output,
  st::Shape shape,
  st::Coefficients coefficients,
  cudaStream_t stream
) {
  constexpr dim3 block = block_shape();
  // Round partial tiles up without losing interiors.
  const dim3 grid(
    cuda::ceil_div(static_cast<unsigned>(shape.width - 2), kOutputTile),
    cuda::ceil_div(static_cast<unsigned>(shape.height - 2), kOutputTile),
    cuda::ceil_div(static_cast<unsigned>(shape.depth - 2), kOutputTile)
  );
  stencil_coarsen<<<grid, block, 0, stream>>>(input, output, shape, coefficients);
}

} // namespace

int main(int argc, char** argv) {
  // The harness checks correctness and benchmarks execution.
  return st::run_app(
    argc, argv,
    {"stencil.coarsen", launch_coarsen, nullptr,
     // Report resources for the 32x32 thread block.
     lg::benchmark::fixed_resources<st::Shape, stencil_coarsen, kInputTile, kInputTile>}
  );
}
