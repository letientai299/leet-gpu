#pragma once

#include "log.hpp"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/std/span>
#include <cuda/stream>
#include <cuda_runtime.h>
#include <fmt/format.h>
#include <source_location>
#include <stdexcept>
#include <type_traits>
#include <utility>

template <typename... Args>
void log_check_failure(
  const char* kind, std::source_location where, fmt::format_string<Args...> format, Args&&... args
) {
  write_log(
    kind, where.file_name(), static_cast<int>(where.line()), format, std::forward<Args>(args)...
  );
}

// Overloaded per status type.
inline bool
check_status(cudaError_t error, std::source_location where = std::source_location::current()) {
  if (error == cudaSuccess) {
    return true;
  }

  log_check_failure("CUDA", where, "{}: {}", cudaGetErrorName(error), cudaGetErrorString(error));
  return false;
}

// Clears stale errors first so a failure belongs to this launch.
#define LAUNCH_CHECK(launch)                                                                       \
  (check_status(cudaGetLastError()) && ((launch), check_status(cudaGetLastError())))

// cudaGetDeviceProperties fills every field and can take milliseconds.
// https://developer.nvidia.com/blog/cuda-pro-tip-the-fast-way-to-query-device-properties/
inline cudaError_t device_attribute(cudaDeviceAttr attribute, int& value) {
  int device = 0;
  const auto status = cudaGetDevice(&device);
  return status != cudaSuccess ? status : cudaDeviceGetAttribute(&value, attribute, device);
}

#ifdef LG_HAS_NPP
#include <nppdefs.h>

inline bool
check_status(NppStatus status, std::source_location where = std::source_location::current()) {
  if (status == NPP_SUCCESS) {
    return true;
  }

  log_check_failure("NPP", where, "status {}", static_cast<int>(status));
  return false;
}

#endif

#ifdef LG_HAS_CUTLASS
#include <cutlass/cutlass.h>

inline bool
check_status(cutlass::Status status, std::source_location where = std::source_location::current()) {
  if (status == cutlass::Status::kSuccess) {
    return true;
  }

  log_check_failure("CUTLASS", where, "{}", cutlass::cutlassGetStatusString(status));
  return false;
}

#endif

#ifdef LG_HAS_CUDA_DRIVER
#include <cuda.h>

inline bool
check_status(CUresult result, std::source_location where = std::source_location::current()) {
  if (result == CUDA_SUCCESS) {
    return true;
  }

  const char* name = nullptr;
  if (cuGetErrorName(result, &name) != CUDA_SUCCESS) {
    name = "unknown";
  }
  log_check_failure("CUDA", where, "driver {}", name);
  return false;
}

#endif

#ifdef LG_HAS_CUPTI
#include <cupti_result.h>

inline bool
check_status(CUptiResult result, std::source_location where = std::source_location::current()) {
  if (result == CUPTI_SUCCESS) {
    return true;
  }

  const char* text = nullptr;
  if (cuptiGetResultString(result, &text) != CUPTI_SUCCESS) {
    text = "unknown";
  }
  log_check_failure("CUPTI", where, "{}", text);
  return false;
}

#endif

#ifdef LG_HAS_CUBLAS
#include <cublas_v2.h>

inline bool
check_status(cublasStatus_t status, std::source_location where = std::source_location::current()) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return true;
  }

  log_check_failure("CUBLAS", where, "{}", cublasGetStatusString(status));
  return false;
}

/// Owns a cuBLAS handle bound to the device current at creation.
class CublasHandle {
public:
  CublasHandle() {
    if (!check_status(cublasCreate(&handle_))) {
      throw std::runtime_error("cuBLAS handle creation failed");
    }
  }
  ~CublasHandle() {
    check_status(cublasDestroy(handle_));
  }
  CublasHandle(const CublasHandle&) = delete;
  CublasHandle& operator=(const CublasHandle&) = delete;

  [[nodiscard]] cublasHandle_t get() const {
    return handle_;
  }

private:
  cublasHandle_t handle_ = nullptr;
};
#endif

inline cuda::stream_ref default_stream() {
  return cuda::stream_ref{cudaStream_t{nullptr}};
}

inline auto& device_pool() {
  int device = 0;
  if (!check_status(cudaGetDevice(&device))) {
    throw std::runtime_error("CUDA device selection failed");
  }
  return cuda::device_default_memory_pool(cuda::devices[device]);
}

// Shared device-to-host readback. cuda::copy_bytes would be the CCCL-idiomatic
// call, but it fails with "invalid argument" on CCCL 3.4.2 + CTK 13.4, so this
// stays on the runtime API.
// https://nvidia.github.io/cccl/libcudacxx/extended_api/algorithms/copy_bytes.html
template <typename Source, typename Destination>
bool copy_checked(
  const Source& source,
  Destination& destination,
  std::source_location where = std::source_location::current()
) {
  using Element = std::remove_pointer_t<decltype(destination.data())>;
  using SourceElement = std::remove_pointer_t<decltype(source.data())>;
  static_assert(std::is_trivially_copyable_v<Element>, "readback needs a trivially copyable type");
  static_assert(std::is_same_v<std::remove_cv_t<Element>, std::remove_cv_t<SourceElement>>);
  if (source.size() > destination.size()) {
    log_check_failure(
      "CUDA", where, "copy destination holds {} of {} elements",
      static_cast<std::size_t>(destination.size()), static_cast<std::size_t>(source.size())
    );
    return false;
  }
  return check_status(
    cudaMemcpy(
      destination.data(), source.data(), source.size() * sizeof(Element), cudaMemcpyDeviceToHost
    ),
    where
  );
}

inline __host__ __device__ void not_implemented(const char* file, int line) {
#ifdef __CUDA_ARCH__
  if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 &&
      threadIdx.y == 0 && threadIdx.z == 0) {
    __assert_fail(
      "kernel is not implemented", file, static_cast<unsigned int>(line), "NOT_IMPLEMENTED"
    );
  }
#else
  write_log("CUDA", file, line, "kernel is not implemented");
  throw std::logic_error("kernel is not implemented");
#endif
}

#define NOT_IMPLEMENTED() not_implemented(__FILE__, __LINE__)

inline bool init_cuda() {
  int device_id = 0;
  cudaDeviceProp device{};
  // cudaFree(nullptr) forces primary-context creation before anything queries it.
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html
  if (!check_status(cudaFree(nullptr)) || !check_status(cudaGetDevice(&device_id)) ||
      !check_status(cudaGetDeviceProperties(&device, device_id))) {
    return false;
  }

  set_gpu_arch(device.major, device.minor);
  HOST_LOG("{} (sm_{}{})", device.name, device.major, device.minor);
  return true;
}

inline bool init_host() {
  init_log();
  return init_cuda();
}

inline bool help_requested(int argc, char** argv) {
  return argc == 2 && std::strcmp(argv[1], "--help") == 0;
}

/// Turns an escaped exception into exit status 1.
template <typename Body> int guarded(Body&& body) try {
  return std::forward<Body>(body)();
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}

template <typename Fn> int run_host(int argc, char** argv, Fn&& body) {
  return guarded([&] {
    if (help_requested(argc, argv)) {
      std::printf("Usage: %s\n", argv[0]);
      return 0;
    }
    return init_host() ? std::forward<Fn>(body)() : 1;
  });
}

inline void log_image_launch(dim3 grid, dim3 block, std::size_t pixels) {
  const auto threads = static_cast<std::size_t>(grid.x) * grid.y * block.x * block.y;
  HOST_LOG(
    "Grid {}x{}, block {}x{}, threads {}, pixels {}", grid.x, grid.y, block.x, block.y, threads,
    pixels
  );
}
