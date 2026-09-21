#pragma once

#include "benchmark/resources.cuh"

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

using Launch = void (*)(
  const float* input, const float* filter, float* output, Shape shape, cudaStream_t stream
);
using FilterSetup = cudaError_t (*)(const float* filter, Shape shape, cudaStream_t stream);

struct Traffic {
  std::size_t global_reads = 0;
  std::size_t global_writes = 0;
};

using TrafficCallback = bool (*)(Shape, Traffic&);
using ResourcesCallback = cudaError_t (*)(const Shape&, lg::benchmark::KernelResources& resources);

inline std::optional<std::size_t> checked_mul(std::size_t left, std::size_t right) {
  if (right != 0 && left > static_cast<std::size_t>(-1) / right) {
    return std::nullopt;
  }
  return left * right;
}

inline std::size_t valid_axis_pairs(int length, int radius) {
  std::size_t pairs = 0;
  for (int position = 0; position < length; ++position) {
    const int first = position > radius ? position - radius : 0;
    const int last = position + radius < length ? position + radius : length - 1;
    pairs += static_cast<std::size_t>(last - first + 1);
  }
  return pairs;
}

inline bool basic_traffic(Shape shape, Traffic& traffic) {
  const auto xy = checked_mul(
    valid_axis_pairs(shape.width, shape.radius), valid_axis_pairs(shape.height, shape.radius)
  );
  const auto taps =
    xy ? checked_mul(*xy, valid_axis_pairs(shape.depth, shape.radius)) : std::nullopt;
  const auto reads = taps ? checked_mul(*taps, std::size_t{2}) : std::nullopt;
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
