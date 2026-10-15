#pragma once

#include <cuda_runtime_api.h>
#include <functional>

namespace lg::reduction {

using Launch =
  std::function<void(float* input, float* output, unsigned int length, cudaStream_t stream)>;

struct Kernel {
  const char* name;
  Launch launch;
  unsigned int segment_size;
  unsigned int max_length; // Zero permits multiple blocks.
};

int run(int argc, const char* const* argv, Kernel kernel);

} // namespace lg::reduction
