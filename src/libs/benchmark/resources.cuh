#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace lg::benchmark {

/// Concrete resources assigned to one compiled kernel launch.
struct KernelResources {
  int threads_per_block = 0;
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0;
  std::size_t dynamic_shared_bytes = 0;
  std::size_t constant_bytes = 0;
  std::size_t local_bytes_per_thread = 0;
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

  resources.threads_per_block = static_cast<int>(block.x * block.y * block.z);
  resources.registers_per_thread = attributes.numRegs;
  resources.static_shared_bytes = attributes.sharedSizeBytes;
  resources.dynamic_shared_bytes = dynamic_shared_bytes;
  resources.constant_bytes = attributes.constSizeBytes;
  resources.local_bytes_per_thread = attributes.localSizeBytes;
  return cudaSuccess;
}

/// Adapts a fixed launch configuration to a shape-aware resource callback.
template <typename Shape,
          auto DeviceKernel,
          unsigned BlockX,
          unsigned BlockY = 1,
          unsigned BlockZ = 1,
          std::size_t DynamicSharedBytes = 0>
cudaError_t fixed_resources(const Shape&, KernelResources& resources) {
  return inspect_kernel<DeviceKernel>(dim3{BlockX, BlockY, BlockZ}, DynamicSharedBytes, resources);
}

} // namespace lg::benchmark
