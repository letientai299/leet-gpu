#pragma once

#include "benchmark/resources.cuh"
#include "benchmark/workload.hpp"
#include "validation.hpp"

#include <cstddef>
#include <cuda_runtime_api.h>
#include <optional>

namespace lg::convolution3d {

struct Shape {
  int width;
  int height;
  int depth;
  int radius;

  [[nodiscard]] __host__ __device__ constexpr int filter_width() const {
    return radius * 2 + 1;
  }
  [[nodiscard]] constexpr std::size_t input_size() const {
    return static_cast<std::size_t>(width) * height * depth;
  }
  [[nodiscard]] constexpr std::size_t filter_size() const {
    const auto side = static_cast<std::size_t>(filter_width());
    return side * side * side;
  }
};

inline void validate_shape(Shape shape) {
  constexpr int max_dimension = std::numeric_limits<int>::max();
  if (shape.width <= 0 || shape.height <= 0 || shape.depth <= 0 || shape.radius < 0 ||
      shape.radius > (max_dimension - 1) / 2 ||
      std::max({shape.width, shape.height, shape.depth}) > max_dimension - 2 * shape.radius) {
    throw std::invalid_argument("invalid convolution dimensions");
  }
  const auto xy = lg::checked_mul(static_cast<std::size_t>(shape.width), shape.height);
  const auto xyz = xy ? lg::checked_mul(*xy, shape.depth) : std::nullopt;
  const auto side = static_cast<std::size_t>(shape.filter_width());
  const auto area = lg::checked_mul(side, side);
  const auto volume = area ? lg::checked_mul(*area, side) : std::nullopt;
  if (!xyz || !volume || !lg::checked_mul(*xyz, sizeof(float)) ||
      !lg::checked_mul(*volume, sizeof(float))) {
    throw std::invalid_argument("convolution size overflow");
  }
}

using Launch = void (*)(
  const float* input, const float* filter, float* output, Shape shape, cudaStream_t stream
);
using FilterSetup = cudaError_t (*)(const float* filter, Shape shape, cudaStream_t stream);

using lg::benchmark::Traffic;

using TrafficCallback = bool (*)(Shape, Traffic&);
using ResourcesCallback = cudaError_t (*)(const Shape&, lg::benchmark::KernelResources& resources);

using lg::checked_mul;
using lg::valid_axis_pairs;

/// Counts in-bounds (output, filter) pairs; each is one multiply-add.
inline std::optional<std::size_t> taps(Shape shape) {
  const auto xy = checked_mul(
    valid_axis_pairs(shape.width, shape.radius), valid_axis_pairs(shape.height, shape.radius)
  );
  return xy ? checked_mul(*xy, valid_axis_pairs(shape.depth, shape.radius)) : std::nullopt;
}

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

} // namespace lg::convolution3d
