#include "convolution3d/benchmark.hpp"

#include <cuda/cmath>

namespace {

namespace conv = lg::convolution3d;

constexpr int kTile = 8;
constexpr int kMaxFilterRadius = 7;
constexpr int kMaxFilterWidth = kMaxFilterRadius * 2 + 1;
constexpr int kMaxInTile = kTile + 2 * kMaxFilterRadius;

__constant__ float gFilter[kMaxFilterWidth * kMaxFilterWidth * kMaxFilterWidth]{};

constexpr dim3 block_shape() {
  return {kTile, kTile, kTile};
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

// PMPP Ex 7.10 3D revision of the §7.4 tiled kernel.
//
// A100X SM80 at 1.215 GHz; 256×256×128, r=3.
// Cold: 1.787 ms; 3134.3 GFLOP/s.
// Batch: 1.779 ms; 3148.3 GFLOP/s.
// NVBench converged: 0.26% cold GPU noise.
__global__ void conv3d_tiled_kernel(const float* input, float* output, conv::Shape shape) {
  __shared__ float tile[kMaxInTile][kMaxInTile][kMaxInTile];

  const int z = static_cast<int>(blockIdx.z * blockDim.z + threadIdx.z);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

  const int tz = static_cast<int>(threadIdx.z);
  const int ty = static_cast<int>(threadIdx.y);
  const int tx = static_cast<int>(threadIdx.x);

  const int in_tile = kTile + 2 * shape.radius;
  const int z0 = z - tz;
  const int y0 = y - ty;
  const int x0 = x - tx;
  for (int i = tz; i < in_tile; i += kTile) {
    for (int j = ty; j < in_tile; j += kTile) {
      for (int k = tx; k < in_tile; k += kTile) {
        const int iz = z0 + i - shape.radius;
        const int iy = y0 + j - shape.radius;
        const int ix = x0 + k - shape.radius;
        tile[i][j][k] = (iz >= 0 && iz < shape.depth &&  //
                         iy >= 0 && iy < shape.height && //
                         ix >= 0 && ix < shape.width)
                          ? input[(iz * shape.height + iy) * shape.width + ix]
                          : 0.0f;
      }
    }
  }
  __syncthreads();

  if (x >= shape.width || y >= shape.height || z >= shape.depth) {
    return;
  }

  const int fw = shape.filter_width();
  float sum = 0.0f;
  for (int fz = 0; fz < fw; ++fz) {
    for (int fy = 0; fy < fw; ++fy) {
      for (int fx = 0; fx < fw; ++fx) {
        const float filter_value = gFilter[(fz * fw + fy) * fw + fx];
        // halo is already in this tile (0 if ghost), so every tap is SMEM.
        sum += tile[tz + fz][ty + fy][tx + fx] * filter_value;
      }
    }
  }

  output[(z * shape.height + y) * shape.width + x] = sum;
}

void launch_tiled(
  const float* input, const float*, float* output, conv::Shape shape, cudaStream_t stream
) {
  constexpr dim3 block = block_shape();
  const dim3 grid(
    cuda::ceil_div(shape.width, block.x), cuda::ceil_div(shape.height, block.y),
    cuda::ceil_div(shape.depth, block.z)
  );
  conv3d_tiled_kernel<<<grid, block, 0, stream>>>(input, output, shape);
}

} // namespace

int main(int argc, char** argv) {
  return conv::run_app(
    argc, argv,
    {"conv3d.tiled", launch_tiled, constant_traffic,
     lg::benchmark::fixed_resources<conv::Shape, conv3d_tiled_kernel, 8, 8, 8>, setup_filter}
  );
}
