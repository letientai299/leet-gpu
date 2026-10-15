#include "benchmark/resources.cuh"
#include "histogram.hpp"

static constexpr unsigned int kThreads = 256;
static constexpr unsigned int kBins = lg::histogram::kBins;

static unsigned int block_count(unsigned int length) {
  return length == 0 ? 1 : (length - 1) / kThreads + 1;
}

// One private copy per block in global memory. Block 0 uses the result.
static unsigned int output_size(unsigned int length) {
  return block_count(length) * kBins;
}

__global__ void histo_private_kernel(const char* data, unsigned int n, unsigned int* hst) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  const auto sliceIdx = blockIdx.x * kBins;

  // work on the private copy of the histogram table, which is a slice of hst.
  if (i < n) {
    const auto pos = data[i] - 'a';
    if (pos >= 0 && pos < 26) {
      atomicAdd(&hst[sliceIdx + pos / 4], 1);
    }
  }

  // block 0 private hst copy is also the output,
  // so, exclude it during this merging phase.
  if (blockIdx.x > 0) {
    __syncthreads();
    // Block-strided loop: the block cooperatively commits kBins counters,
    // and kBins is independent of blockDim.x.
    // - Thread t takes bins t, t + blockDim.x, t + 2 * blockDim.x, ...
    //   Each bin has exactly one owner, so no two threads add the same
    //   slice counter. Neighbor threads read neighbor addresses, so the
    //   slice reads coalesce.
    // - kBins > blockDim.x: required. A plain `threadIdx.x < kBins` check
    //   drops bins >= blockDim.x. E.g. 1024 bins, 256 threads: 4 bins/thread.
    // - kBins <= blockDim.x (here 7 <= 256): the loop runs at most once per
    //   thread, and threads >= kBins skip it. Equivalent to the plain check.
    // The stride keeps the kernel correct if kBins or kThreads change.
    for (auto bin = threadIdx.x; bin < kBins; bin += blockDim.x) {
      const auto v = hst[sliceIdx + bin];
      if (v > 0) {
        atomicAdd(&hst[bin], v);
      }
    }
  }
}

static void
launch(const char* data, unsigned int length, unsigned int* histo, cudaStream_t stream) {
  if (length == 0) {
    return;
  }
  histo_private_kernel<<<block_count(length), kThreads, 0, stream>>>(data, length, histo);
}

static cudaError_t resources(lg::benchmark::KernelResources& result) {
  return lg::benchmark::inspect_kernel<histo_private_kernel>(dim3{kThreads}, 0, result);
}

int main(int argc, char** argv) {
  return lg::histogram::run(argc, argv, launch, resources, output_size);
}
