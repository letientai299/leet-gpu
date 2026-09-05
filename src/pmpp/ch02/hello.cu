#include "device.hpp"

namespace {

constexpr int blocks = 3;
constexpr int threads = 5;

__global__ void hello_kernel(int* output) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  output[index] = (static_cast<int>(blockIdx.x) + 1) * 100 + static_cast<int>(threadIdx.x) + 1;
}

int run_hello() {
  int output[blocks * threads]{};
  DeviceBuffer<int> device_output;
  if (!device_output.allocate(sizeof(output))) {
    return 1;
  }

  hello_kernel<<<blocks, threads>>>(device_output.get());
  if (!CUDA_CHECK(cudaGetLastError()) ||
      !CUDA_CHECK(
          cudaMemcpy(output, device_output.get(), sizeof(output), cudaMemcpyDeviceToHost))) {
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
