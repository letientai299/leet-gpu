#include "reduction.hpp"

namespace {

constexpr unsigned int kThreads = 256;
constexpr unsigned int kSegmentSize = 2 * kThreads;

// PMPP 4e, §10.3, Fig. 10.6.
__global__ void reduce_basic(float* in, float* out) {
  const auto i = threadIdx.x * 2; // we have only 1 block
  const auto n = blockDim.x;      // the block dimension is 1/2 the input size.
  for (auto stride = 1; stride <= n; stride *= 2) {
    if (threadIdx.x % stride == 0) {
      in[i] += in[i + stride];
    }
    __syncthreads();
  }

  if (i == 0) {
    *out = in[0];
  }
}

void launch(float* in, float* out, unsigned int /*length*/, cudaStream_t stream) {
  reduce_basic<<<1, kThreads, 0, stream>>>(in, out);
}

} // namespace

int main(int argc, char** argv) {
  return lg::reduction::run(argc, argv, {"reduce.basic", launch, kSegmentSize, kSegmentSize});
}
