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
#include <stdexcept>
#include <type_traits>
#include <utility>

inline bool cuda_check(cudaError_t error, const char* file, int line) {
  if (error == cudaSuccess) {
    return true;
  }

  write_log("CUDA", file, line, "%s: %s", cudaGetErrorName(error), cudaGetErrorString(error));
  return false;
}

#define CUDA_CHECK(expression) cuda_check((expression), __FILE__, __LINE__)

#ifdef LG_HAS_NPP
#include <nppdefs.h>

inline bool npp_check(NppStatus status, const char* file, int line) {
  if (status == NPP_SUCCESS) {
    return true;
  }

  write_log("NPP", file, line, "status %d", static_cast<int>(status));
  return false;
}

#define NPP_CHECK(expression) npp_check((expression), __FILE__, __LINE__)
#endif

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

// Shared device-to-host readback. cuda::copy_bytes would be the CCCL-idiomatic
// call, but it fails with "invalid argument" on CCCL 3.4.2 + CTK 13.4, so this
// stays on the runtime API.
// https://nvidia.github.io/cccl/libcudacxx/extended_api/algorithms/copy_bytes.html
template <typename Source, typename Destination>
bool copy_checked(const Source& source, Destination& destination, const char* file, int line) {
  using Element = std::remove_pointer_t<decltype(destination.data())>;
  static_assert(std::is_trivially_copyable_v<Element>, "readback needs a trivially copyable type");
  if (source.size() > destination.size()) {
    write_log("CUDA", file, line, "copy destination holds %zu of %zu elements",
              static_cast<std::size_t>(destination.size()),
              static_cast<std::size_t>(source.size()));
    return false;
  }
  return cuda_check(cudaMemcpy(destination.data(), source.data(), source.size() * sizeof(Element),
                               cudaMemcpyDeviceToHost),
                    file, line);
}

#define COPY_CHECK(source, destination) copy_checked((source), (destination), __FILE__, __LINE__)

inline __host__ __device__ void not_implemented(const char* file, int line) {
#ifdef __CUDA_ARCH__
  if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 &&
      threadIdx.y == 0 && threadIdx.z == 0) {
    __assert_fail("kernel is not implemented", file, static_cast<unsigned int>(line),
                  "NOT_IMPLEMENTED");
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
  if (!CUDA_CHECK(cudaFree(nullptr)) || !CUDA_CHECK(cudaGetDevice(&device_id)) ||
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
