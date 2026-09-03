#include "cuda_check.hpp"

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
  if (!CUDA_CHECK(cudaMalloc(&device_output, sizeof(output)))) {
    return false;
  }

  fp4_kernel<<<1, 1>>>(1.5F, device_output);
  const bool ok = CUDA_CHECK(cudaGetLastError()) &&
                  CUDA_CHECK(cudaMemcpy(&output, device_output, sizeof(output),
                                        cudaMemcpyDeviceToHost));
  cudaFree(device_output);
  if (!ok) {
    return false;
  }

  GPU_LOG("FP4 round trip: %.1f", output);
  return output == 1.5F;
}
