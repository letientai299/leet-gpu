#include "reduction.hpp"

#include <cstddef>

namespace {

constexpr unsigned int kThreads = 256;
constexpr unsigned int kCoarseFactor = 4;
constexpr unsigned int kSegmentSize = kCoarseFactor * 2 * kThreads;

// PMPP 4e, Ex 10.5.
__global__ void reduce_coarse(const float* in, float* out, unsigned int n) {
  __shared__ float shared[kThreads];
  const auto segment = static_cast<std::size_t>(blockIdx.x) * kSegmentSize;
  const auto t = threadIdx.x;
  const auto i = segment + t;

  float sum = 0;
  for (auto c = 0; c < kCoarseFactor; c++) {
    const auto offset = i + static_cast<std::size_t>(c * 2 * blockDim.x);
    if (offset < n) {
      sum += in[offset];
    }
    if (offset + blockDim.x < n) {
      sum += in[offset + blockDim.x];
    }
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
  if (n == 0) {
    return;
  }
  const auto blocks = (n - 1) / kSegmentSize + 1;
  reduce_coarse<<<blocks, kThreads, 0, stream>>>(in, out, n);
}

} // namespace

int main(int argc, char* const* argv) {
  return lg::reduction::run(
    argc, argv, {"reduce.coarse.arbitrary", launch, kSegmentSize, 0, false}
  );
}
