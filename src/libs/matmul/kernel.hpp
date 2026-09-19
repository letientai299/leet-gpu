#pragma once

#include "matmul/matrix.hpp"

#include <cstddef>
#include <cuda_runtime_api.h>
#include <limits>
#include <optional>
#include <string_view>

namespace lg::matmul {

using KernelCallback = void (*)(const float*, const float*, float*, const GemmShape&, cudaStream_t);

/// Saturating helpers: counts derived from a shape can exceed size_t.
inline std::optional<std::size_t> checked_mul(std::size_t left, std::size_t right) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
    return std::nullopt;
  }
  return left * right;
}

inline std::optional<std::size_t> checked_add(std::size_t left, std::size_t right) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    return std::nullopt;
  }
  return left + right;
}

/// Bytes a kernel moves through global memory, for NVBench bandwidth columns.
struct Traffic {
  std::size_t global_reads = 0;
  std::size_t global_writes = 0;
};

using TrafficCallback = bool (*)(const GemmShape&, Traffic&);

/// One thread per output element: each reads a full A row and B column.
inline bool cell_traffic(const GemmShape& shape, Traffic& traffic) {
  const auto reads = checked_mul(shape.c_size(), std::size_t{shape.k()} * 2);
  if (!reads) {
    return false;
  }
  traffic.global_reads = *reads;
  traffic.global_writes = shape.c_size();
  return true;
}

struct Kernel {
  std::string_view name;
  KernelCallback launch;
  TrafficCallback traffic = nullptr;
};

} // namespace lg::matmul
