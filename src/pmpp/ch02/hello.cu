#include "checks.hpp"

#include <cstddef>
#include <cuda/buffer>

namespace {

constexpr int blocks = 3;
constexpr int threads = 5;

__global__ void hello_kernel(int* output) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  output[index] = (static_cast<int>(blockIdx.x) + 1) * 100 + static_cast<int>(threadIdx.x) + 1;
}

int run_hello() {
  int output[blocks * threads]{};
  constexpr auto count = static_cast<std::size_t>(blocks) * threads;
  cuda::device_buffer<int> device_output{default_stream(), device_pool(), count, cuda::no_init};

  hello_kernel<<<blocks, threads>>>(device_output.data());
  if (!CUDA_CHECK(cudaGetLastError()) ||
      !CUDA_CHECK(
          cudaMemcpy(output, device_output.data(), sizeof(output), cudaMemcpyDeviceToHost))) {
    return 1;
  }

  for (const int value : output) {
    GPU_LOG("Hello from block %d, thread %d", value / 100, value % 100);
  }
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  return run_host(argc, argv, run_hello);
}
