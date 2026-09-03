#include "cuda_check.hpp"

namespace {

__global__ __cluster_dims__(2, 1, 1) void cluster_kernel(int *output) {
  if (threadIdx.x == 0) {
    output[blockIdx.x] = static_cast<int>(blockIdx.x) + 10;
  }
  asm volatile("barrier.cluster.arrive;" ::: "memory");
  asm volatile("barrier.cluster.wait;" ::: "memory");
  if (threadIdx.x == 0) {
    output[blockIdx.x] += 1;
  }
}

} // namespace

bool run_hopper_cluster() {
  int *device_output = nullptr;
  int output[2]{};
  if (!CUDA_CHECK(cudaMalloc(&device_output, sizeof(output)))) {
    return false;
  }

  cluster_kernel<<<2, 32>>>(device_output);
  const bool ok = CUDA_CHECK(cudaGetLastError()) &&
                  CUDA_CHECK(cudaMemcpy(output, device_output, sizeof(output),
                                        cudaMemcpyDeviceToHost));
  cudaFree(device_output);
  if (!ok) {
    return false;
  }

  GPU_LOG("cluster barrier: %d %d", output[0], output[1]);
  return output[0] == 11 && output[1] == 12;
}
