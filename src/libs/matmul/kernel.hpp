#pragma once

#include "matmul/matrix.hpp"

#include <cuda_runtime_api.h>
#include <limits>
#include <string_view>

namespace lg::matmul {

using KernelCallback = void (*)(const float*, const float*, float*, const GemmShape&, cudaStream_t);

struct Traffic {
  std::size_t global_reads = 0;
  std::size_t global_writes = 0;
};

using TrafficCallback = bool (*)(const GemmShape&, Traffic&);

inline bool cell_traffic(const GemmShape& shape, Traffic& traffic) {
  const std::size_t outputs = shape.c_size();
  if (outputs > std::numeric_limits<std::size_t>::max() / shape.k() / 2) {
    return false;
  }
  traffic.global_reads = outputs * shape.k() * 2;
  traffic.global_writes = outputs;
  return true;
}

struct Kernel {
  std::string_view name;
  KernelCallback launch;
  TrafficCallback traffic = nullptr;
};

} // namespace lg::matmul
