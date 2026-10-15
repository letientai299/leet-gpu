#include "reduction.hpp"

namespace {

constexpr unsigned int kThreads = 256;
constexpr unsigned int kSegmentSize = 2 * kThreads;
constexpr unsigned int kDim = kThreads;

// PMPP 4e, §10.6, Fig. 10.11.
__global__ void reduce_shared(const float* in, float* out) {
  __shared__ float shared[kDim];
  const auto n = blockDim.x;
  const auto t = threadIdx.x;

  shared[t] = in[t] + in[t + kDim];

  for (auto stride = n / 2; stride > 0; stride /= 2) {
    __syncthreads();
    if (t < stride) {
      shared[t] += shared[t + stride];
    }
  }

  if (t == 0) {
    *out = shared[0];
  }
}

void launch(const float* in, float* out, unsigned int /*n*/, cudaStream_t stream) {
  reduce_shared<<<1, kThreads, 0, stream>>>(in, out);
}

} // namespace

int main(int argc, char* const* argv) {
  return lg::reduction::run(argc, argv, {"reduce.shared", launch, kSegmentSize, kSegmentSize});
}
