#include "benchmark/resources.cuh"
#include "histogram.hpp"

static constexpr unsigned int kThreads = 256;

__global__ void histo_kernel(const char* data, unsigned int n, unsigned int* hst) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const auto pos = data[i] - 'a';
    if (pos >= 0 && pos < 26) {
      atomicAdd(&hst[pos / 4], 1);
    }
  }
}

static void
launch(const char* data, unsigned int length, unsigned int* histo, cudaStream_t stream) {
  if (length == 0) {
    return;
  }
  const unsigned int blocks = (length - 1) / kThreads + 1;
  histo_kernel<<<blocks, kThreads, 0, stream>>>(data, length, histo);
}

static cudaError_t resources(lg::benchmark::KernelResources& result) {
  return lg::benchmark::inspect_kernel<histo_kernel>(dim3{kThreads}, 0, result);
}

int main(int argc, char** argv) {
  return lg::histogram::run(argc, argv, launch, resources);
}
