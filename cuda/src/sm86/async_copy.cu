#include "cuda_check.hpp"

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cstdio>

namespace cg = cooperative_groups;

namespace {

constexpr int count = 32;

__global__ void async_copy_kernel(const int *input, int *output) {
  __shared__ int staged[count];
  auto block = cg::this_thread_block();
  cg::memcpy_async(block, staged, input, sizeof(staged));
  cg::wait(block);
  output[threadIdx.x] = staged[threadIdx.x] * 2;
}

} // namespace

bool run_ampere_async_copy() {
  int input[count];
  for (int i = 0; i < count; ++i) {
    input[i] = i;
  }

  int *device_input = nullptr;
  int *device_output = nullptr;
  int output[count]{};
  if (!cuda_check(cudaMalloc(&device_input, sizeof(input))) ||
      !cuda_check(cudaMalloc(&device_output, sizeof(output))) ||
      !cuda_check(cudaMemcpy(device_input, input, sizeof(input),
                             cudaMemcpyHostToDevice))) {
    cudaFree(device_input);
    cudaFree(device_output);
    return false;
  }

  async_copy_kernel<<<1, count>>>(device_input, device_output);
  const bool ok = cuda_check(cudaGetLastError()) &&
                  cuda_check(cudaMemcpy(output, device_output, sizeof(output),
                                        cudaMemcpyDeviceToHost));
  cudaFree(device_input);
  cudaFree(device_output);
  if (!ok) {
    return false;
  }

  printf("[GPU ] cp.async result: %d\n", output[count - 1]);
  return output[count - 1] == (count - 1) * 2;
}
