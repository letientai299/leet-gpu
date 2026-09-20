#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

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

using Launch = void (*)(
    const float* input, const float* filter, float* output, Shape shape, cudaStream_t stream);

struct Kernel {
  const char* name;
  Launch launch;
};

} // namespace lg::convolution
