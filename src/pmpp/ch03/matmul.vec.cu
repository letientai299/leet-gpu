#include "checks.hpp"
#include "matmul/correctness.hpp"

#include <cuda/cmath>

namespace {

namespace mm = lg::matmul;

// Ex 3.2: A[i] = sum_j B[i][j] * C[j]. One thread per output element.
// a: output vector n, b: row-major n*n matrix, c: input vector n.
__global__ void matvec_kernel(float* a, const float* b, const float* c, unsigned n) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float sum = 0;
    for (auto j = 0; j < n; j++) {
      sum += c[j] * b[i * n + j];
    }
    a[i] = sum;
  }
}

/// Host stub: one thread per output vector element.
void matvec(float* a, const float* b, const float* c, unsigned n, cudaStream_t stream) {
  constexpr unsigned block = 256;
  const auto grid = static_cast<unsigned>(cuda::ceil_div(n, block));
  matvec_kernel<<<grid, block, 0, stream>>>(a, b, c, n);
}

void launch_matvec(const float* matrix,
                   const float* vec_in,
                   float* vec_out,
                   const mm::GemmShape& shape,
                   cudaStream_t stream) {
  matvec(vec_out, matrix, vec_in, shape.m(), stream);
}

int run_matvec() {
  // Square GEMM with one output column.
  constexpr unsigned n = 67;
  mm::Problem problem(mm::GemmShape(n, 1, n));
  mm::fill_random(problem);
  return mm::check(problem, {"matmul.vec", launch_matvec});
}

} // namespace

int main(int argc, char** argv) {
  return run_host(argc, argv, run_matvec);
}
