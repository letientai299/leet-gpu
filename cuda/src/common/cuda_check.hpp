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
