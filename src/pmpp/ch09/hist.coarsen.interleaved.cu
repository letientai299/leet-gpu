#include "benchmark/resources.cuh"
#include "histogram.hpp"

static constexpr unsigned int kThreads = 256;
static constexpr unsigned int kBins = lg::histogram::kBins;
static constexpr unsigned int cFactor = 32; // coarsening factor

static unsigned int block_count(unsigned int length) {
  return length == 0 ? 1 : (length - 1) / (cFactor * kThreads) + 1;
}

__global__ void
histo_coarsen_interleaved_kernel(const char* data, unsigned int n, unsigned int* hst) {
  // init privatized bins
  __shared__ unsigned int shared[kBins];
  if (threadIdx.x < kBins) {
    shared[threadIdx.x] = 0;
  }
  __syncthreads();

  // Compute histogram on the shared mem
  // const auto base = blockIdx.x * blockDim.x * cFactor;
  // for (unsigned int c = 0; c < cFactor; c++) {
  //   const auto i = base + c * blockDim.x + threadIdx.x;
  //   if (i < n) {
  //     const auto pos = data[i] - 'a';
  //     if (pos >= 0 && pos < 26) {
  //       atomicAdd(&shared[pos / 4], 1);
  //     }
  //   }
  // }

  // PMPP 4e, Fig. 9.14
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto stride = blockDim.x * gridDim.x;
  for (unsigned int i = tid; i < n; i += stride) {
    const auto pos = data[i] - 'a';
    if (pos >= 0 && pos < 26) {
      atomicAdd(&shared[pos / 4], 1);
    }
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
  histo_coarsen_interleaved_kernel<<<block_count(length), kThreads, 0, stream>>>(
    data, length, histo
  );
}

static cudaError_t resources(lg::benchmark::KernelResources& result) {
  return lg::benchmark::inspect_kernel<histo_coarsen_interleaved_kernel>(dim3{kThreads}, 0, result);
}

int main(int argc, char** argv) {
  return lg::histogram::run(argc, argv, launch, resources);
}
