#pragma once

#include "log.hpp"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/stream>
#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>

inline bool cuda_check(cudaError_t error, const char* file, int line) {
  if (error == cudaSuccess) {
    return true;
  }

  write_log("CUDA", file, line, "%s: %s", cudaGetErrorName(error), cudaGetErrorString(error));
  return false;
}

#define CUDA_CHECK(expression) cuda_check((expression), __FILE__, __LINE__)

#ifdef LG_HAS_CUTLASS
#include <cutlass/cutlass.h>

inline bool cutlass_check(cutlass::Status status, const char* file, int line) {
  if (status == cutlass::Status::kSuccess) {
    return true;
  }

  write_log("CUTLASS", file, line, "%s", cutlass::cutlassGetStatusString(status));
  return false;
}

#define CUTLASS_CHECK(expression) cutlass_check((expression), __FILE__, __LINE__)
#endif

inline cuda::stream_ref default_stream() {
  return cuda::stream_ref{cudaStream_t{nullptr}};
}

inline auto& device_pool() {
  return cuda::device_default_memory_pool(cuda::devices[0]);
}

[[noreturn]] inline __host__ __device__ void not_implemented() {
#ifdef __CUDA_ARCH__
  __trap();
#else
  throw std::logic_error("not implemented");
#endif
}

inline bool init_cuda() {
  int device_id = 0;
  cudaDeviceProp device{};
  if (!CUDA_CHECK(cudaGetDevice(&device_id)) || !CUDA_CHECK(cudaFree(nullptr)) ||
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

inline void log_image_launch(dim3 grid, dim3 block, std::size_t pixels) {
  const auto threads = static_cast<std::size_t>(grid.x) * grid.y * block.x * block.y;
  HOST_LOG("Grid %ux%u, block %ux%u, threads %zu, pixels %zu", grid.x, grid.y, block.x, block.y,
           threads, pixels);
}
