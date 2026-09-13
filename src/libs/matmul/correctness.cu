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

template <typename Operation>
bool on_device(Problem& problem, Matrix& output, Operation&& operation) {
  const auto stream = default_stream();
  auto& pool = device_pool();
  const DeviceMatrix device_a{stream, pool, problem.a};
  const DeviceMatrix device_b{stream, pool, problem.b};
  DeviceMatrix device_c{stream, pool, output};
  if (!operation(device_a, device_b, device_c)) {
    return false;
  }
  return CUDA_CHECK(
      cudaMemcpy(output.data(), device_c.data(), output.bytes(), cudaMemcpyDeviceToHost));
}

} // namespace

bool run_reference(Problem& problem) {
  using RowMajor = cutlass::layout::RowMajor;
  using Gemm = cutlass::gemm::device::Gemm<float, RowMajor, float, RowMajor, float, RowMajor>;

  return on_device(problem, problem.expected,
                   [&](const DeviceMatrix& a, const DeviceMatrix& b, DeviceMatrix& c) {
                     constexpr float alpha = 1.0F;
                     constexpr float beta = 0.0F;
                     const int m = static_cast<int>(problem.shape.m());
                     const int n = static_cast<int>(problem.shape.n());
                     const int k = static_cast<int>(problem.shape.k());
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
  std::fill(problem.result.begin(), problem.result.end(), 0.0F);
  return on_device(problem, problem.result,
                   [&](const DeviceMatrix& a, const DeviceMatrix& b, DeviceMatrix& c) {
                     kernel.launch(a.data(), b.data(), c.data(), problem.shape, nullptr);
                     return CUDA_CHECK(cudaGetLastError());
                   });
}

bool verify(const Problem& problem) {
  for (std::size_t index = 0; index < problem.result.size(); ++index) {
    const float actual = problem.result.data()[index];
    const float expected = problem.expected.data()[index];
    const float diff = std::fabs(actual - expected);
    const float tolerance = kAbsoluteTolerance + kRelativeTolerance * std::fabs(expected);
    if (!std::isfinite(actual) || diff > tolerance) {
      HOST_LOG("Mismatch at %zu: %g vs %g (diff %g, tol %g)", index, actual, expected, diff,
               tolerance);
      return false;
    }
  }
  HOST_LOG("Matrix multiply passed: %zu values", problem.result.size());
  return true;
}

int check(Problem& problem, Kernel kernel) {
  return run_reference(problem) && run_kernel(problem, kernel) && verify(problem) ? 0 : 1;
}

} // namespace lg::matmul
