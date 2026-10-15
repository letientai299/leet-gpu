#pragma once

#include <cuda_runtime.h>
#include <functional>
#include <optional>

namespace lg::benchmark {

struct HardwareMetrics {
  double occupancy = 0;
  double dram_read_bytes = 0;
  double dram_write_bytes = 0;
  double dram_bytes_per_second = 0;
  double dram_utilization = 0;
  double dram_peak = 0;
  double fp32_flops = 0;
  double fp32_flops_per_second = 0;
  double fp32_peak = 0;
  double fp64_flops = 0;
  double fp64_flops_per_second = 0;
  double fp64_peak = 0;
  double fp16_utilization = 0;
  double tensor_utilization = 0;
  double integer_instructions = 0;
  double integer_instructions_per_second = 0;
  double integer_utilization = 0;
  double global_atomics = 0;
  double global_reductions = 0;
  double shared_atomics = 0;
  double shared_atomic_utilization = 0;
  double atomic_sectors = 0;
  double atomic_sectors_per_second = 0;
  double atomic_utilization = 0;
  double reduction_sectors = 0;
  double reduction_sectors_per_second = 0;
  double reduction_utilization = 0;
};

/// Collects counters separately from unprofiled timing.
std::optional<HardwareMetrics>
profile_hardware(int device, cudaStream_t stream, const std::function<void(cudaStream_t)>& launch);

} // namespace lg::benchmark
