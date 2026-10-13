#pragma once

#include <cuda_runtime.h>
#include <functional>
#include <optional>

namespace lg::benchmark {

/// Profiles one `launch` with CUPTI; returns active warps / SM maximum.
std::optional<double> achieved_occupancy(
  int device, cudaStream_t stream, const std::function<void(cudaStream_t)>& launch
);

} // namespace lg::benchmark
