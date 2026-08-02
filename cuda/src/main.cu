#include <cstdio>

__global__ static void hello_kernel() {
  printf("[GPU ] Hello from block %d, thread %d\n", blockIdx.x + 1,
         threadIdx.x + 1);
}

int main() {
  printf("[Host] CUDA ======= \n");
  hello_kernel<<<3, 5>>>();
  cudaDeviceSynchronize();
  printf("[Host] Done ======= \n");
  return 0;
}
