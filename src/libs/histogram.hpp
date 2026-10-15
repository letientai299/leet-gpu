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

int run(int argc, char** argv, Launch launch, Resources resources);

} // namespace lg::histogram
