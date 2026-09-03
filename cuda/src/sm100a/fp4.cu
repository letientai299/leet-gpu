#include "cuda_check.hpp"

#include <cstdio>
#include <cuda_fp4.h>

namespace {

__global__ void fp4_kernel(float input, float *output) {
  const __nv_fp4_e2m1 value(input);
  *output = static_cast<float>(value);
}

} // namespace

bool run_blackwell_fp4() {
  float *device_output = nullptr;
  float output = 0.0F;
  if (!cuda_check(cudaMalloc(&device_output, sizeof(output)))) {
    return false;
  }

  fp4_kernel<<<1, 1>>>(1.5F, device_output);
  const bool ok = cuda_check(cudaGetLastError()) &&
                  cuda_check(cudaMemcpy(&output, device_output, sizeof(output),
                                        cudaMemcpyDeviceToHost));
  cudaFree(device_output);
  if (!ok) {
    return false;
  }

  printf("[GPU ] FP4 round trip: %.1f\n", output);
  return output == 1.5F;
}
