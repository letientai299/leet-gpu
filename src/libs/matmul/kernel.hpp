#pragma once

#include "matmul/matrix.hpp"

#include <cuda_runtime_api.h>
#include <string_view>

namespace lg::matmul {

using KernelCallback = void (*)(const float*, const float*, float*, const GemmShape&, cudaStream_t);

struct Kernel {
  std::string_view name;
  KernelCallback launch;
};

} // namespace lg::matmul
