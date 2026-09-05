#pragma once

#include "log.hpp"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cublas_v2.h>
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

inline bool cublas_check(cublasStatus_t status, const char* file, int line) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return true;
  }

  write_log("CUBLAS", file, line, "%s", cublasGetStatusString(status));
  return false;
}

#define CUBLAS_CHECK(expression) cublas_check((expression), __FILE__, __LINE__)

class CublasHandle {
public:
  CublasHandle() = default;
  CublasHandle(const CublasHandle&) = delete;
  CublasHandle& operator=(const CublasHandle&) = delete;

  ~CublasHandle() {
    if (handle_ != nullptr) {
      cublasDestroy(handle_);
    }
  }

  bool create() {
    // Pedantic IEEE FP32: default math may use TF32 on Ampere+ and miss a naive kernel.
    return CUBLAS_CHECK(cublasCreate(&handle_)) &&
           CUBLAS_CHECK(cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH));
  }

  [[nodiscard]] cublasHandle_t get() const {
    return handle_;
  }

private:
  cublasHandle_t handle_ = nullptr;
};

inline cuda::stream_ref default_stream() {
  return cuda::stream_ref{cudaStream_t{nullptr}};
}

inline auto& device_pool() {
  return cuda::device_default_memory_pool(default_stream().device());
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

inline void log_image_launch(dim3 grid, dim3 block, std::size_t pixels) {
  const auto threads = static_cast<std::size_t>(grid.x) * grid.y * block.x * block.y;
  HOST_LOG("Grid %ux%u, block %ux%u, threads %zu, pixels %zu", grid.x, grid.y, block.x, block.y,
           threads, pixels);
}
