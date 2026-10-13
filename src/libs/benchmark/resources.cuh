#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <limits>

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

/// Queries launch resources without executing the kernel.
template <auto DeviceKernel>
cudaError_t
inspect_kernel(dim3 block, std::size_t dynamic_shared_bytes, KernelResources& resources) {
  cudaFuncAttributes attributes{};
  const cudaError_t result = cudaFuncGetAttributes(&attributes, DeviceKernel);
  if (result != cudaSuccess) {
    return result;
  }

  int device = 0;
  cudaDeviceProp properties{};
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }
  status = cudaGetDeviceProperties(&properties, device);
  if (status != cudaSuccess) {
    return status;
  }
  if (block.x == 0 || block.y == 0 || block.z == 0 ||
      block.x > static_cast<unsigned>(properties.maxThreadsDim[0]) ||
      block.y > static_cast<unsigned>(properties.maxThreadsDim[1]) ||
      block.z > static_cast<unsigned>(properties.maxThreadsDim[2])) {
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
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &resources.active_blocks_per_sm, DeviceKernel, resources.threads_per_block, dynamic_shared_bytes
  );
  resources.active_warps_per_sm =
    resources.active_blocks_per_sm *
    ((resources.threads_per_block + properties.warpSize - 1) / properties.warpSize);
  resources.max_warps_per_sm = properties.maxThreadsPerMultiProcessor / properties.warpSize;
  return status;
}

/// Adapts a fixed launch configuration to a shape-aware resource callback.
template <
  typename Shape,
  auto DeviceKernel,
  unsigned BlockX,
  unsigned BlockY = 1,
  unsigned BlockZ = 1,
  std::size_t DynamicSharedBytes = 0>
cudaError_t fixed_resources(const Shape&, KernelResources& resources) {
  return inspect_kernel<DeviceKernel>(dim3{BlockX, BlockY, BlockZ}, DynamicSharedBytes, resources);
}

} // namespace lg::benchmark
