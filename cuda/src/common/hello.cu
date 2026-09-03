#include "cuda_check.hpp"

#include <cstdio>

namespace {

__global__ void hello_kernel() {
  printf("[GPU ] Hello from block %d, thread %d\n", blockIdx.x + 1,
         threadIdx.x + 1);
}

} // namespace

bool run_hello() {
  hello_kernel<<<3, 5>>>();
  return cuda_check(cudaGetLastError()) && cuda_check(cudaDeviceSynchronize());
}
