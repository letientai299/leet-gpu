#pragma once

#include "checks.hpp"

#include <cstddef>

template <typename T> class DeviceBuffer {
public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    cudaFree(ptr_);
  }

  bool allocate(std::size_t bytes) {
    void* raw = nullptr;
    if (!CUDA_CHECK(cudaMalloc(&raw, bytes))) {
      return false;
    }
    ptr_ = static_cast<T*>(raw);
    return true;
  }

  [[nodiscard]] T* get() const {
    return ptr_;
  }

private:
  T* ptr_ = nullptr;
};
