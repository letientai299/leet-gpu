#include <cstdio>

__global__ static void hello_kernel() {
  const int i = 9;
  printf("[GPU ] Hello from block %d, thread %d, i=%d\n", blockIdx.x + 1,
         threadIdx.x + 1, i);
}

int main() {
  printf("[Host] CUDA ======= \n");
  hello_kernel<<<3, 5>>>();
  cudaDeviceSynchronize();
  printf("[Host] Done ======= \n");
  return 0;
}
