#include "reduction.hpp"

namespace {

constexpr unsigned int kThreads = 256;
constexpr unsigned int kSegmentSize = 2 * kThreads;
constexpr unsigned int kDim = kThreads;

// PMPP 4e, §10.7, Fig. 10.13.
__global__ void reduce_hierarchical(const float* in, float* out) {
  __shared__ float shared[kDim];
  const auto segment = 2 * blockDim.x * blockIdx.x;
  const auto t = threadIdx.x;
  const auto i = segment + t;
  shared[t] = in[i] + in[i + blockDim.x];

  for (auto stride = blockDim.x / 2; stride > 0; stride /= 2) {
    __syncthreads();
    if (t < stride) {
      shared[t] += shared[t + stride];
    }
  }

  if (t == 0) {
    atomicAdd(out, shared[0]);
  }
}

void launch(const float* in, float* out, unsigned int n, cudaStream_t stream) {
  reduce_hierarchical<<<n / kSegmentSize, kThreads, 0, stream>>>(in, out);
}

} // namespace

int main(int argc, char* const* argv) {
  return lg::reduction::run(argc, argv, {"reduce.hierarchical", launch, kSegmentSize, 0});
}
