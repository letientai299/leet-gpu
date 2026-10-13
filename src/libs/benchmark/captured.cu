#include "benchmark/captured.hpp"
#include "checks.hpp"
#include "log.hpp"

#include <cstddef>
#include <vector>

namespace lg::benchmark {
namespace {

/// Mirrors `cudaFuncGetAttributes` for a driver handle.
bool get_attributes(CUfunction function, cudaFuncAttributes& attributes) {
  int shared = 0;
  int constant = 0;
  int local = 0;
  const bool ok =
    DRIVER_CHECK(cuFuncGetAttribute(&attributes.numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, function)) &&
    DRIVER_CHECK(cuFuncGetAttribute(
      &attributes.maxThreadsPerBlock, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function
    )) &&
    DRIVER_CHECK(cuFuncGetAttribute(&shared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function)) &&
    DRIVER_CHECK(cuFuncGetAttribute(&constant, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, function)) &&
    DRIVER_CHECK(cuFuncGetAttribute(&local, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function));
  attributes.sharedSizeBytes = static_cast<std::size_t>(shared);
  attributes.constSizeBytes = static_cast<std::size_t>(constant);
  attributes.localSizeBytes = static_cast<std::size_t>(local);
  return ok;
}

/// Checks that a step of the chain has one node; 0 ends it.
cudaError_t chain_step(std::size_t count) {
  if (count > 1) {
    HOST_LOG("Captured graph branches; expected one stream");
    return cudaErrorNotSupported;
  }
  return cudaSuccess;
}

/// Lists nodes in launch order. One-stream capture yields a chain;
/// `cudaGraphGetNodes` order is unspecified.
cudaError_t launch_order(cudaGraph_t graph, std::vector<cudaGraphNode_t>& order) {
  std::size_t count = 0;
  auto status = cudaGraphGetRootNodes(graph, nullptr, &count);
  if (status == cudaSuccess) {
    status = count == 0 ? cudaErrorInvalidValue : chain_step(count);
  }
  cudaGraphNode_t node = nullptr;
  if (status == cudaSuccess) {
    status = cudaGraphGetRootNodes(graph, &node, &count);
  }
  while (status == cudaSuccess) {
    order.push_back(node);
    status = cudaGraphNodeGetDependentNodes(node, nullptr, nullptr, &count);
    if (status != cudaSuccess || count == 0) {
      break;
    }
    status = chain_step(count);
    if (status == cudaSuccess) {
      status = cudaGraphNodeGetDependentNodes(node, &node, nullptr, &count);
    }
  }
  return status;
}

} // namespace

cudaError_t inspect_graph(cudaGraph_t graph, KernelResources& resources) {
  std::vector<cudaGraphNode_t> nodes;
  auto status = launch_order(graph, nodes);
  if (status != cudaSuccess) {
    return status;
  }

  std::vector<CUDA_KERNEL_NODE_PARAMS> kernels;
  for (auto* node : nodes) {
    cudaGraphNodeType type{};
    status = cudaGraphNodeGetType(node, &type);
    if (status != cudaSuccess) {
      return status;
    }
    if (type != cudaGraphNodeTypeKernel) {
      continue;
    }
    // Library launches go through the driver, so read driver params.
    CUDA_KERNEL_NODE_PARAMS params{};
    if (!DRIVER_CHECK(cuGraphKernelNodeGetParams(node, &params)) ||
        (params.func == nullptr && !DRIVER_CHECK(cuKernelGetFunction(&params.func, params.kern)))) {
      return cudaErrorUnknown;
    }
    const char* name = nullptr;
    if (!DRIVER_CHECK(cuFuncGetName(&name, params.func))) {
      name = "unknown";
    }
    HOST_LOG(
      "Captured kernel %s grid %ux%ux%u block %ux%ux%u dyn %u B", name, params.gridDimX,
      params.gridDimY, params.gridDimZ, params.blockDimX, params.blockDimY, params.blockDimZ,
      params.sharedMemBytes
    );
    kernels.push_back(params);
  }
  if (kernels.empty()) {
    return cudaErrorInvalidValue;
  }
  if (kernels.size() > 1) {
    HOST_LOG("Resources show the first launched of %zu kernels", kernels.size());
  }

  const auto& first = kernels.front();
  cudaFuncAttributes attributes{};
  if (!get_attributes(first.func, attributes)) {
    return cudaErrorUnknown;
  }
  const auto occupancy = [&first](int threads, int& blocks) {
    return DRIVER_CHECK(cuOccupancyMaxActiveBlocksPerMultiprocessor(
             &blocks, first.func, threads, first.sharedMemBytes
           ))
             ? cudaSuccess
             : cudaErrorUnknown;
  };
  return fill_resources(
    attributes, dim3{first.blockDimX, first.blockDimY, first.blockDimZ}, first.sharedMemBytes,
    occupancy, resources
  );
}

} // namespace lg::benchmark
