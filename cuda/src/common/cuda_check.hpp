#pragma once

#include <cstdio>
#include <cuda_runtime.h>

inline bool cuda_check(cudaError_t error) {
  if (error == cudaSuccess) {
    return true;
  }

  fprintf(stderr, "CUDA error: %s: %s\n", cudaGetErrorName(error),
          cudaGetErrorString(error));
  return false;
}
