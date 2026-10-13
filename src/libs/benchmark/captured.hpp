#pragma once

#include "benchmark/resources.cuh"

#include <cuda/devices>
#include <cuda/stream>
#include <cuda_runtime.h>

namespace lg::benchmark {

/// Reads resources of the first kernel launched in a captured graph.
cudaError_t inspect_graph(cudaGraph_t graph, KernelResources& resources);

/// Captures `launch(stream)` without running it, then inspects its kernels.
/// Library calls hide their kernels, so `inspect_kernel` cannot name them.
template <typename Launch> cudaError_t inspect_launch(Launch&& launch, KernelResources& resources) {
  int device = 0;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }
  cuda::stream const stream{cuda::devices[device]};
  status = cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal);
  if (status != cudaSuccess) {
    return status;
  }
  cudaGraph_t graph = nullptr;
  try {
    launch(stream.get());
  } catch (...) {
    // Leave capture so the stream can be destroyed.
    if (cudaStreamEndCapture(stream.get(), &graph) == cudaSuccess && graph != nullptr) {
      cudaGraphDestroy(graph);
    }
    throw;
  }
  status = cudaStreamEndCapture(stream.get(), &graph);
  if (status != cudaSuccess) {
    return status;
  }
  status = inspect_graph(graph, resources);
  const auto destroyed = cudaGraphDestroy(graph);
  return status != cudaSuccess ? status : destroyed;
}

} // namespace lg::benchmark
