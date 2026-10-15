#include "reduction.hpp"

namespace {

constexpr unsigned int kThreads = 256;
constexpr unsigned int kCoarseFactor = 4;
constexpr unsigned int kSegmentSize = kCoarseFactor * 2 * kThreads;
constexpr unsigned int kDim = kThreads;

// PMPP 4e, §10.8, Fig. 10.15.
__global__ void reduce_coarse(const float* in, float* out) {
  __shared__ float shared[kDim];
  const auto segment = 2 * blockDim.x * blockIdx.x * kCoarseFactor;
  const auto t = threadIdx.x;
  const auto i = segment + t;

  float sum = 0;
  for (auto c = 0; c < kCoarseFactor; c++) {
    const auto offset = i + c * 2 * blockDim.x;
    sum += in[offset] + in[offset + blockDim.x];
  }
  shared[t] = sum;

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
  reduce_coarse<<<n / kSegmentSize, kThreads, 0, stream>>>(in, out);
}

} // namespace

int main(int argc, char* const* argv) {
  return lg::reduction::run(argc, argv, {"reduce.coarse", launch, kSegmentSize, 0});
}
