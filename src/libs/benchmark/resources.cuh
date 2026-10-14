#pragma once

#include "checks.hpp"

#include <cstddef>
#include <cuda_runtime.h>
#include <utility>

namespace lg::benchmark {

/// Concrete resources assigned to one compiled kernel launch.
struct KernelResources {
  dim3 block{0, 0, 0};
  int threads_per_block = 0;
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0;
  std::size_t dynamic_shared_bytes = 0;
  std::size_t constant_bytes = 0;
  std::size_t local_bytes_per_thread = 0;
  int active_blocks_per_sm = 0;
  int active_warps_per_sm = 0;
  int max_warps_per_sm = 0;
};

/// Fills resources from compiled attributes; `occupancy` reports blocks/SM.
template <typename Occupancy>
cudaError_t fill_resources(
  const cudaFuncAttributes& attributes,
  dim3 block,
  std::size_t dynamic_shared_bytes,
  Occupancy&& occupancy,
  KernelResources& resources
) {
  int max_x = 0;
  int max_y = 0;
  int max_z = 0;
  int warp_size = 0;
  int max_sm_threads = 0;
  for (const auto& [attribute, value] :
       {std::pair{cudaDevAttrMaxBlockDimX, &max_x},
        {cudaDevAttrMaxBlockDimY, &max_y},
        {cudaDevAttrMaxBlockDimZ, &max_z},
        {cudaDevAttrWarpSize, &warp_size},
        {cudaDevAttrMaxThreadsPerMultiProcessor, &max_sm_threads}}) {
    if (const auto status = device_attribute(attribute, *value); status != cudaSuccess) {
      return status;
    }
  }
  if (block.x == 0 || block.y == 0 || block.z == 0 || block.x > static_cast<unsigned>(max_x) ||
      block.y > static_cast<unsigned>(max_y) || block.z > static_cast<unsigned>(max_z)) {
    return cudaErrorInvalidConfiguration;
  }
  const auto threads = static_cast<std::size_t>(block.x) * block.y * block.z;
  if (threads > static_cast<std::size_t>(attributes.maxThreadsPerBlock)) {
    return cudaErrorInvalidConfiguration;
  }
  resources.block = block;
  resources.threads_per_block = static_cast<int>(threads);
  resources.registers_per_thread = attributes.numRegs;
  resources.static_shared_bytes = attributes.sharedSizeBytes;
  resources.dynamic_shared_bytes = dynamic_shared_bytes;
  resources.constant_bytes = attributes.constSizeBytes;
  resources.local_bytes_per_thread = attributes.localSizeBytes;
  const auto status = occupancy(resources.threads_per_block, resources.active_blocks_per_sm);
  resources.active_warps_per_sm =
    resources.active_blocks_per_sm * ((resources.threads_per_block + warp_size - 1) / warp_size);
  resources.max_warps_per_sm = max_sm_threads / warp_size;
  return status;
}

/// Queries launch resources without executing the kernel.
template <auto DeviceKernel>
cudaError_t
inspect_kernel(dim3 block, std::size_t dynamic_shared_bytes, KernelResources& resources) {
  cudaFuncAttributes attributes{};
  const cudaError_t result = cudaFuncGetAttributes(&attributes, DeviceKernel);
  if (result != cudaSuccess) {
    return result;
  }
  const auto occupancy = [dynamic_shared_bytes](int threads, int& blocks) {
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, DeviceKernel, threads, dynamic_shared_bytes
    );
  };
  return fill_resources(attributes, block, dynamic_shared_bytes, occupancy, resources);
}

/// Adapts a fixed launch configuration to a shape-aware resource callback.
template <typename Shape, auto DeviceKernel, dim3 Block, std::size_t DynamicSharedBytes = 0>
cudaError_t fixed_resources(const Shape&, KernelResources& resources) {
  return inspect_kernel<DeviceKernel>(Block, DynamicSharedBytes, resources);
}

} // namespace lg::benchmark
