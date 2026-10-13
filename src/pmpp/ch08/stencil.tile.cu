#include "stencil/benchmark.hpp"

#include <cstddef>
#include <cuda/cmath>

namespace {

namespace st = lg::stencil;

// One halo layer per face leaves 6^3 output points from 8^3 input threads.
// Halo threads load neighbors but do not compute outputs.
constexpr unsigned kInputTile = 8;
constexpr unsigned kOutputTile = kInputTile - 2;

constexpr dim3 block_shape() {
  return {kInputTile, kInputTile, kInputTile};
}

// PMPP §8.3 shared memory tiling (Fig. 8.8).
__global__ void stencil_tile(const float* in, float* out, st::Shape sh, st::Coefficients co) {
  // Along each axis, block b loads [6b, 6b+7] and writes [6b+1, 6b+6].
  // Input tiles overlap by two cells; output tiles never overlap.
  // The first block's lower halo is grid coordinate 0, so no -1 shift is needed.
  const int x = static_cast<int>(blockIdx.x * kOutputTile + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * kOutputTile + threadIdx.y);
  const int z = static_cast<int>(blockIdx.z * kOutputTile + threadIdx.z);
  const unsigned tx = threadIdx.x;
  const unsigned ty = threadIdx.y;
  const unsigned tz = threadIdx.z;

  // Keep x contiguous in shared memory, matching the global grid layout.
  __shared__ float tile[kInputTile][kInputTile][kInputTile];

  const std::size_t index =
    static_cast<std::size_t>(z) * sh.plane_size() + static_cast<std::size_t>(y) * sh.width + x;
  // Partial tiles leave out-of-grid cells unloaded. A valid output's six
  // neighbors are all in-grid, so computation never reads those cells.
  if (x < sh.width && y < sh.height && z < sh.depth) {
    tile[tz][ty][tx] = in[index];
  }

  // Outputs consume values loaded by other threads. Even halo and out-of-grid
  // threads reach this barrier; output-only returns belong after it.
  __syncthreads();

  // Preserve the grid boundary: the harness initializes it from the input.
  const bool grid_interior = x >= 1 && x < sh.width - 1 &&  //
                             y >= 1 && y < sh.height - 1 && //
                             z >= 1 && z < sh.depth - 1;

  // Exclude tile halos so each output has exactly one writer. Grid-interior
  // alone would allow overlapping input tiles to write the same output.
  const bool tile_interior = tx >= 1 && tx < kInputTile - 1 && //
                             ty >= 1 && ty < kInputTile - 1 && //
                             tz >= 1 && tz < kInputTile - 1;

  if (!grid_interior || !tile_interior) {
    return;
  }

  // The seven-point stencil uses the center and six face neighbors only.
  // This simple cubic load also fetches unused tile edges and corners.
  out[index] = co.c0 * tile[tz][ty][tx] +     //
               co.c1 * tile[tz][ty][tx - 1] + //
               co.c2 * tile[tz][ty][tx + 1] + //
               co.c3 * tile[tz][ty - 1][tx] + //
               co.c4 * tile[tz][ty + 1][tx] + //
               co.c5 * tile[tz - 1][ty][tx] + //
               co.c6 * tile[tz + 1][ty][tx];
}

void launch_tile(
  const float* input,
  float* output,
  st::Shape shape,
  st::Coefficients coefficients,
  cudaStream_t stream
) {
  constexpr dim3 block = block_shape();
  // Only shape-2 points per axis need updates. Divide by the output width, 6,
  // rather than the block width, 8; round up to cover the final partial tile.
  const dim3 grid(
    cuda::ceil_div(static_cast<unsigned>(shape.width - 2), kOutputTile),
    cuda::ceil_div(static_cast<unsigned>(shape.height - 2), kOutputTile),
    cuda::ceil_div(static_cast<unsigned>(shape.depth - 2), kOutputTile)
  );
  stencil_tile<<<grid, block, 0, stream>>>(input, output, shape, coefficients);
}

} // namespace

int main(int argc, char** argv) {
  // The harness checks outputs and benchmarks this launch. nullptr leaves
  // traffic accounting unset; basic_traffic would miscount these tiled reads.
  return st::run_app(
    argc, argv,
    {"stencil.tile", launch_tile, nullptr,
     lg::benchmark::fixed_resources<st::Shape, stencil_tile, kInputTile, kInputTile, kInputTile>}
  );
}
