#pragma once

#include "log.hpp"

#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <utility>

inline bool cuda_check(cudaError_t error, const char* file, int line) {
  if (error == cudaSuccess) {
    return true;
  }

  write_log("CUDA", file, line, "%s: %s", cudaGetErrorName(error), cudaGetErrorString(error));
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

inline bool init_host() {
  init_log();
  return init_cuda();
}

inline bool help_requested(int argc, char** argv) {
  return argc == 2 && std::strcmp(argv[1], "--help") == 0;
}

template <typename Fn> int run_host(int argc, char** argv, Fn&& body) {
  if (help_requested(argc, argv)) {
    std::printf("Usage: %s\n", argv[0]);
    return 0;
  }
  if (!init_host()) {
    return 1;
  }
  return std::forward<Fn>(body)();
}
