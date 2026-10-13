#pragma once

#include "convolution/kernel.hpp"

namespace lg::convolution::constant {

constexpr int kMaxFilterRadius = 7;
constexpr int kMaxFilterWidth = kMaxFilterRadius * 2 + 1;

// global var in the constant memory for storing the filter array
static __constant__ float gFilter[kMaxFilterWidth * kMaxFilterWidth]{};

inline bool traffic(Shape shape, Traffic& traffic) {
  const auto reads = taps(shape);
  if (!reads) {
    return false;
  }
  traffic.global_reads = *reads;
  traffic.global_writes = shape.input_size();
  return true;
}

inline cudaError_t setup_filter(const float* filter, Shape shape, cudaStream_t stream) {
  if (shape.radius < 0 || shape.radius > kMaxFilterRadius) {
    return cudaErrorInvalidValue;
  }
  return cudaMemcpyToSymbolAsync(
    gFilter, filter, shape.filter_size() * sizeof(float), 0, cudaMemcpyDeviceToDevice, stream
  );
}

} // namespace lg::convolution::constant
