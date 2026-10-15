#include "benchmark/resources.cuh"
#include "histogram.hpp"

static constexpr unsigned int kThreads = 256;
static constexpr unsigned int kBins = lg::histogram::kBins;
static constexpr unsigned int cFactor = 32; // coarsening factor

static unsigned int block_count(unsigned int length) {
  return length == 0 ? 1 : (length - 1) / (cFactor * kThreads) + 1;
}

__global__ void histo_aggregate_v2_kernel(const char* data, unsigned int n, unsigned int* hst) {
  // init privatized bins
  __shared__ unsigned int shared[kBins];
  if (threadIdx.x < kBins) {
    shared[threadIdx.x] = 0;
  }
  __syncthreads();

  // Issue all loads before any atomic, as in interleaved v2.
  const auto base = blockIdx.x * blockDim.x * cFactor;
  char chars[cFactor];

  for (unsigned int c = 0; c < cFactor; c++) {
    const auto i = base + c * blockDim.x + threadIdx.x;
    chars[c] = i < n ? data[i] : 0;
  }

  // Compute histogram on the shared mem
  // PMPP 4e, Fig. 9.15
  unsigned int acc = 0;
  int prevBinIdx = 0;
  // Without unroll, chars spills to local memory.
#pragma unroll
  for (const auto ch : chars) {
    const auto pos = ch - 'a';
    if (pos >= 0 && pos < 26) {
      const auto bin = pos / 4;
      if (bin == prevBinIdx) {
        acc++;
      } else {
        // Flush the run only when the bin changes.
        if (acc > 0) {
          atomicAdd(&shared[prevBinIdx], acc);
        }
        acc = 1;
        prevBinIdx = bin;
      }
    }
  }
  if (acc > 0) {
    atomicAdd(&shared[prevBinIdx], acc);
  }
  __syncthreads();

  // Commit to global mem
  if (threadIdx.x < kBins && shared[threadIdx.x] > 0) {
    atomicAdd(&hst[threadIdx.x], shared[threadIdx.x]);
  }
}

static void
launch(const char* data, unsigned int length, unsigned int* histo, cudaStream_t stream) {
  if (length == 0) {
    return;
  }
  histo_aggregate_v2_kernel<<<block_count(length), kThreads, 0, stream>>>(data, length, histo);
}

static cudaError_t resources(lg::benchmark::KernelResources& result) {
  return lg::benchmark::inspect_kernel<histo_aggregate_v2_kernel>(dim3{kThreads}, 0, result);
}

int main(int argc, char** argv) {
  return lg::histogram::run(argc, argv, launch, resources);
}
