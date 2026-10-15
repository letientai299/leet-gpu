#pragma once

#include <cuda_runtime_api.h>

namespace lg::benchmark {
struct KernelResources;
}

namespace lg::histogram {

inline constexpr unsigned int kBins = 7; // ceil(26/4)

using Launch =
  void (*)(const char* data, unsigned int length, unsigned int* histo, cudaStream_t stream);

using Resources = cudaError_t (*)(lg::benchmark::KernelResources&);

// Counters the kernel needs for `length` bytes. The first kBins hold the result.
using OutputSize = unsigned int (*)(unsigned int length);

inline unsigned int shared_output(unsigned int) {
  return kBins;
}

int run(
  int argc, char** argv, Launch launch, Resources resources, OutputSize output_size = shared_output
);

} // namespace lg::histogram
