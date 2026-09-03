#include <cstdio>
#include <cuda_runtime.h>

static bool check(cudaError_t error) {
  if (error == cudaSuccess) {
    return true;
  }

  fprintf(stderr, "CUDA error: %s: %s\n", cudaGetErrorName(error),
          cudaGetErrorString(error));
  return false;
}

__global__ static void hello_kernel() {
  const int i = 9;
  printf("[GPU ] Hello from block %d, thread %d, i=%d\n", blockIdx.x + 1,
         threadIdx.x + 1, i);
}

int main() {
  printf("[Host] CUDA ======= \n");
  hello_kernel<<<3, 5>>>();
  if (!check(cudaGetLastError()) || !check(cudaDeviceSynchronize())) {
    return 1;
  }
  printf("[Host] Done ======= \n");
  return 0;
}
