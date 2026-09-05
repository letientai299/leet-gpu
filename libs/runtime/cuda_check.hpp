#pragma once

#include "log.hpp"

#include <cuda_runtime.h>

inline bool cuda_check(cudaError_t error, const char *file, int line) {
  if (error == cudaSuccess) {
    return true;
  }

  write_log("CUDA", file, line, "%s: %s", cudaGetErrorName(error),
            cudaGetErrorString(error));
  return false;
}

#define CUDA_CHECK(expression) cuda_check((expression), __FILE__, __LINE__)

inline bool init_cuda() {
  int device_id = 0;
  cudaDeviceProp device{};
  if (!CUDA_CHECK(cudaGetDevice(&device_id)) ||
      !CUDA_CHECK(cudaGetDeviceProperties(&device, device_id))) {
    return false;
  }

  set_gpu_arch(device.major, device.minor);
  HOST_LOG("%s (sm_%d%d)", device.name, device.major, device.minor);
  return true;
}
