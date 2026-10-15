#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace lg::benchmark {

class ColdCache {
public:
  ColdCache();
  ~ColdCache();
  ColdCache(const ColdCache&) = delete;
  ColdCache& operator=(const ColdCache&) = delete;

  void flush(cudaStream_t stream);

private:
  unsigned int* buffer_ = nullptr;
  std::size_t count_ = 0;
  bool initialized_ = false;
};

} // namespace lg::benchmark
