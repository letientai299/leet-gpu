#pragma once

#include "benchmark/resources.cuh"
#include "benchmark/workload.hpp"
#include "validation.hpp"

#include <cstddef>
#include <cuda_runtime_api.h>
#include <optional>

namespace lg::convolution {

struct Shape {
  int width;
  int height;
  int radius;

  [[nodiscard]] __host__ __device__ constexpr int filter_width() const {
    return radius * 2 + 1;
  }
  [[nodiscard]] constexpr std::size_t input_size() const {
    return static_cast<std::size_t>(width) * height;
  }
  [[nodiscard]] constexpr std::size_t filter_size() const {
    const auto side = static_cast<std::size_t>(filter_width());
    return side * side;
  }
};

inline void validate_shape(Shape shape) {
  constexpr int max_pitch = std::numeric_limits<int>::max() / sizeof(float);
  if (shape.width <= 0 || shape.height <= 0 || shape.radius < 0 ||
      shape.radius > (max_pitch - 1) / 2 || shape.width > max_pitch - 2 * shape.radius ||
      shape.height > std::numeric_limits<int>::max() - 2 * shape.radius) {
    throw std::invalid_argument("invalid convolution dimensions");
  }
}

using Launch = void (*)(
  const float* input, const float* filter, float* output, Shape shape, cudaStream_t stream
);

using FilterSetup = cudaError_t (*)(const float* filter, Shape shape, cudaStream_t stream);

using lg::benchmark::Traffic;

using TrafficCallback = bool (*)(Shape, Traffic&);

using ResourcesCallback = cudaError_t (*)(const Shape&, lg::benchmark::KernelResources& resources);

/// Multiplies counts without overflow.
using lg::checked_mul;

/// Counts in-bounds filter positions along one axis.
using lg::valid_axis_pairs;

/// Counts in-bounds (output, filter) pairs; each is one multiply-add.
inline std::optional<std::size_t> taps(Shape shape) {
  return checked_mul(
    valid_axis_pairs(shape.width, shape.radius), valid_axis_pairs(shape.height, shape.radius)
  );
}

/// Estimates the basic kernel's logical GMEM traffic.
inline bool basic_traffic(Shape shape, Traffic& traffic) {
  const auto count = taps(shape);
  const auto reads = count ? checked_mul(*count, std::size_t{2}) : std::nullopt;
  if (!reads) {
    return false;
  }
  traffic.global_reads = *reads;
  traffic.global_writes = shape.input_size();
  return true;
}

struct Kernel {
  const char* name;
  Launch launch;
  TrafficCallback traffic = nullptr;
  ResourcesCallback resources = nullptr;
  FilterSetup filter_setup = nullptr;
};

} // namespace lg::convolution
