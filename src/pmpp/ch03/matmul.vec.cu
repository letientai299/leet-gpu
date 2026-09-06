#include "matmul.hpp"

#include <cuda/cmath>

namespace {

// Ex 3.2: A[i] = sum_j B[i][j] * C[j]. One thread per output element.
// a: output vector n, b: row-major n*n matrix, c: input vector n.
__global__ void matvec_kernel(float* a, const float* b, const float* c, unsigned n) {
}

/// Host stub: one thread per output vector element.
void matvec(float* a, const float* b, const float* c, unsigned n) {
  constexpr unsigned block = 256;
  const auto grid = static_cast<unsigned>(cuda::ceil_div(n, block));
  matvec_kernel<<<grid, block>>>(a, b, c, n);
}

int run_matvec() {
  // Square GEMM with width 1. Not a multiple of 256: kernel must bound-check.
  constexpr unsigned n = 67;
  return Matmul(n, 1, n).run(
      [](const dbuf& matrix, const dbuf& vec_in, dbuf& vec_out, unsigned n, unsigned, unsigned) {
        matvec(vec_out.data(), matrix.data(), vec_in.data(), n);
        return CUDA_CHECK(cudaGetLastError());
      });
}

} // namespace

int main(int argc, char** argv) {
  return run_host(argc, argv, run_matvec);
}
