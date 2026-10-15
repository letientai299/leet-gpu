#include "reduction.hpp"

namespace {

constexpr unsigned int kThreads = 256;
constexpr unsigned int kSegmentSize = 2 * kThreads;

// PMPP 4e, §10.4, Fig. 10.9.
__global__ void reduce_convergent(float* in, float* out) {
  const auto n = blockDim.x;
  for (auto stride = n; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      in[threadIdx.x] += in[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *out = in[0];
  }
}

void launch(float* in, float* out, unsigned int /*n*/, cudaStream_t stream) {
  reduce_convergent<<<1, kThreads, 0, stream>>>(in, out);
}

} // namespace

int main(int argc, char** argv) {
  return lg::reduction::run(argc, argv, {"reduce.convergent", launch, kSegmentSize, kSegmentSize});
}
