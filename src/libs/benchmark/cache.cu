#include "benchmark/cache.hpp"

#include <nvbench/cuda_call.cuh>

namespace lg::benchmark {
namespace {

__global__ void read_cache(const unsigned int* buffer, std::size_t count) {
  const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
       index += stride) {
    // Keep eviction loads observable.
    asm volatile("{ .reg .u32 value; ld.global.cg.u32 value, [%0]; }"
                 :
                 : "l"(buffer + index)
                 : "memory");
  }
}

__global__ void discard_cache(const unsigned int* buffer, std::size_t count) {
  constexpr std::size_t words = 128 / sizeof(unsigned int);
  const auto stride = static_cast<std::size_t>(gridDim.x) * blockDim.x * words;
  for (auto index = (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * words;
       index < count; index += stride) {
    // Only scratch contents may be discarded.
    asm volatile("discard.global.L2 [%0], 128;" : : "l"(buffer + index) : "memory");
  }
}

} // namespace

ColdCache::ColdCache() {
  int device = 0;
  int bytes = 0;
  NVBENCH_CUDA_CALL(cudaGetDevice(&device));
  NVBENCH_CUDA_CALL(cudaDeviceGetAttribute(&bytes, cudaDevAttrL2CacheSize, device));
  if (bytes > 0) {
    count_ = 2 * static_cast<std::size_t>(bytes) / sizeof(unsigned int);
    NVBENCH_CUDA_CALL(cudaMalloc(&buffer_, count_ * sizeof(unsigned int)));
  }
}

ColdCache::~ColdCache() {
  if (buffer_ != nullptr) {
    NVBENCH_CUDA_CALL_NOEXCEPT(cudaFree(buffer_));
  }
}

void ColdCache::flush(cudaStream_t stream) {
  if (count_ == 0) {
    return;
  }
  if (!initialized_) {
    NVBENCH_CUDA_CALL(cudaMemsetAsync(buffer_, 0, count_ * sizeof(unsigned int), stream));
    initialized_ = true;
  }
  read_cache<<<1024, 256, 0, stream>>>(buffer_, count_);
  NVBENCH_CUDA_CALL(cudaGetLastError());
  discard_cache<<<1024, 256, 0, stream>>>(buffer_, count_);
  NVBENCH_CUDA_CALL(cudaGetLastError());
}

} // namespace lg::benchmark
