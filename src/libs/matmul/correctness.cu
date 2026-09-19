#include "checks.hpp"
#include "matmul/correctness.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cuda/buffer>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>

namespace lg::matmul {
namespace {

using DeviceMatrix = cuda::device_buffer<float>;

/// Uploads A and B, runs `operation`, reads the result back into `output`.
template <typename Operation>
bool on_device(const Matrix& a, const Matrix& b, Matrix& output, Operation&& operation) {
  const auto stream = default_stream();
  auto& pool = device_pool();
  const DeviceMatrix device_a{stream, pool, a};
  const DeviceMatrix device_b{stream, pool, b};
  DeviceMatrix device_c{stream, pool, output};
  return operation(device_a, device_b, device_c) && COPY_CHECK(device_c, output);
}

} // namespace

bool run_reference(Problem& problem) {
  using RowMajor = cutlass::layout::RowMajor;
  using Gemm = cutlass::gemm::device::Gemm<float, RowMajor, float, RowMajor, float, RowMajor>;

  return on_device(problem.a, problem.b, problem.expected,
                   [&](const DeviceMatrix& a, const DeviceMatrix& b, DeviceMatrix& c) {
                     constexpr float alpha = 1.0F;
                     constexpr float beta = 0.0F;
                     const int m = static_cast<int>(problem.shape.m());
                     const int n = static_cast<int>(problem.shape.n());
                     const int k = static_cast<int>(problem.shape.k());
                     // Row-major leading dimensions: lda = k, ldb = ldc = n.
                     const Gemm::Arguments args({m, n, k}, {a.data(), k}, {b.data(), n},
                                                {c.data(), n}, {c.data(), n}, {alpha, beta});
#ifdef __clang_analyzer__
                     (void)args;
                     return true;
#else
                     Gemm gemm;
                     return CUTLASS_CHECK(gemm(args));
#endif
                   });
}

bool run_kernel(Problem& problem, Kernel kernel) {
  if (kernel.launch == nullptr) {
    HOST_LOG("Kernel callback is null");
    return false;
  }
  // Zero first so a kernel that skips elements shows up as a mismatch.
  std::fill(problem.result.begin(), problem.result.end(), 0.0F);
  const KernelInputs inputs(problem, kernel.inputs);
  return on_device(inputs.a(), inputs.b(), problem.result,
                   [&](const DeviceMatrix& a, const DeviceMatrix& b, DeviceMatrix& c) {
                     cudaGetLastError(); // Drop any error left over from an earlier launch.
                     kernel.launch(a.data(), b.data(), c.data(), problem.shape, nullptr);
                     return CUDA_CHECK(cudaGetLastError());
                   });
}

bool verify(const Problem& problem) {
  const Matrix& result = problem.result;
  for (std::size_t index = 0; index < result.size(); ++index) {
    const float actual = result.data()[index];
    const float expected = problem.expected.data()[index];
    const float diff = std::fabs(actual - expected);
    const float tolerance = kAbsoluteTolerance + kRelativeTolerance * std::fabs(expected);
    if (!std::isfinite(actual) || diff > tolerance) {
      HOST_LOG("Mismatch at [%zu,%zu]: %g vs %g (diff %g, tol %g)", index / result.cols(),
               index % result.cols(), actual, expected, diff, tolerance);
      return false;
    }
  }
  HOST_LOG("Matrix multiply passed: %zu values", result.size());
  return true;
}

int check(Problem& problem, Kernel kernel) {
  return run_reference(problem) && run_kernel(problem, kernel) && verify(problem) ? 0 : 1;
}

} // namespace lg::matmul
