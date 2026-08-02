#include <cstdio>

__global__ static void hello_kernel() {
    printf("[GPU ] Hello from block %d, thread %d\n", blockIdx.x+1, threadIdx.x+1);
}

int main() {
    printf("[Host] 17 CUDA ======= \n");
    hello_kernel<<<2, 5>>>();
    cudaDeviceSynchronize();
    printf("[Host] Done 18 ======= \n");
    return 0;
}
