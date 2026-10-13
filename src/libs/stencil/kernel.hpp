#pragma once

#include "benchmark/resources.cuh"
#include "benchmark/workload.hpp"
#include "validation.hpp"

#include <cstddef>
#include <cuda_runtime_api.h>
#include <limits>
#include <optional>
#include <stdexcept>

namespace lg::stencil {

// Row-major grid: x is contiguous, z is slowest.
struct Shape {
  int width;
  int height;
  int depth;

  [[nodiscard]] __host__ __device__ constexpr std::size_t plane_size() const {
    return static_cast<std::size_t>(width) * height;
  }
  [[nodiscard]] __host__ __device__ constexpr std::size_t grid_size() const {
    return plane_size() * depth;
  }
  [[nodiscard]] constexpr std::size_t interior_size() const {
    return static_cast<std::size_t>(width - 2) * (height - 2) * (depth - 2);
  }
};

// PMPP Fig. 8.6 order: c0 center, c1/c2 x, c3/c4 y, c5/c6 z.
struct Coefficients {
  float c0;
  float c1;
  float c2;
  float c3;
  float c4;
  float c5;
  float c6;
};

inline void validate_shape(Shape shape) {
  if (shape.width < 3 || shape.height < 3 || shape.depth < 3) {
    throw std::invalid_argument("stencil grid needs an interior");
  }
  const auto xy = lg::checked_mul(static_cast<std::size_t>(shape.width), shape.height);
  const auto xyz = xy ? lg::checked_mul(*xy, shape.depth) : std::nullopt;
  if (!xyz || *xyz > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      !lg::checked_mul(*xyz, sizeof(float))) {
    throw std::invalid_argument("stencil size overflow");
  }
}

using Launch = void (*)(
  const float* input, float* output, Shape shape, Coefficients coefficients, cudaStream_t stream
);

using lg::benchmark::Traffic;

using TrafficCallback = bool (*)(Shape, Traffic&);
using ResourcesCallback = cudaError_t (*)(const Shape&, lg::benchmark::KernelResources& resources);

inline constexpr std::size_t kPoints = 7;

inline bool basic_traffic(Shape shape, Traffic& traffic) {
  const auto reads = lg::checked_mul(shape.interior_size(), kPoints);
  if (!reads) {
    return false;
  }
  traffic.global_reads = *reads;
  traffic.global_writes = shape.interior_size();
  return true;
}

struct Kernel {
  const char* name;
  Launch launch;
  TrafficCallback traffic = nullptr;
  ResourcesCallback resources = nullptr;
};

} // namespace lg::stencil
