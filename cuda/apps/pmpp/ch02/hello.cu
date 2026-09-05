#include "cuda_check.hpp"

#include <cstdio>
#include <cstring>

namespace {

constexpr int blocks = 3;
constexpr int threads = 5;

__global__ void hello_kernel(int *output) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  output[index] = (static_cast<int>(blockIdx.x) + 1) * 100 +
                  static_cast<int>(threadIdx.x) + 1;
}

void print_help(const char *program) { std::printf("Usage: %s\n", program); }

} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
    print_help(argv[0]);
    return 0;
  }
  if (argc != 1) {
    print_help(argv[0]);
    return 2;
  }

  init_log();
  if (!init_cuda()) {
    return 1;
  }

  int *device_output = nullptr;
  int output[blocks * threads]{};
  if (!CUDA_CHECK(cudaMalloc(&device_output, sizeof(output)))) {
    return 1;
  }

  hello_kernel<<<blocks, threads>>>(device_output);
  const bool ok = CUDA_CHECK(cudaGetLastError()) &&
                  CUDA_CHECK(cudaMemcpy(output, device_output, sizeof(output),
                                        cudaMemcpyDeviceToHost));
  cudaFree(device_output);
  if (!ok) {
    return 1;
  }

  for (const int value : output) {
    GPU_LOG("Hello from block %d, thread %d", value / 100, value % 100);
  }
  return 0;
}
