#include "checks.hpp"
#include "matmul/correctness.hpp"
#include "validation.hpp"

#include <algorithm>
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
  if (problem.a.rows() != problem.shape.m() || problem.a.cols() != problem.shape.k() ||
      problem.b.rows() != problem.shape.k() || problem.b.cols() != problem.shape.n() ||
      problem.expected.rows() != problem.shape.m() ||
      problem.expected.cols() != problem.shape.n() || problem.result.rows() != problem.shape.m() ||
      problem.result.cols() != problem.shape.n()) {
    throw std::invalid_argument("matrix shape mismatch");
  }
  std::fill(
    problem.expected.begin(), problem.expected.end(), std::numeric_limits<float>::quiet_NaN()
  );
  using RowMajor = cutlass::layout::RowMajor;
  using Gemm = cutlass::gemm::device::Gemm<float, RowMajor, float, RowMajor, float, RowMajor>;

  return on_device(
    problem.a, problem.b, problem.expected,
    [&](const DeviceMatrix& a, const DeviceMatrix& b, DeviceMatrix& c) {
      constexpr float alpha = 1.0F;
      constexpr float beta = 0.0F;
      const int m = static_cast<int>(problem.shape.m());
      const int n = static_cast<int>(problem.shape.n());
      const int k = static_cast<int>(problem.shape.k());
      // Row-major leading dimensions: lda = k, ldb = ldc = n.
      const Gemm::Arguments args(
        {m, n, k}, {a.data(), k}, {b.data(), n}, {c.data(), n}, {c.data(), n}, {alpha, beta}
      );
#ifdef __clang_analyzer__
      (void)args;
      return true;
#else
      Gemm gemm;
      return CUTLASS_CHECK(gemm(args));
#endif
    }
  );
}

bool run_kernel(Problem& problem, Kernel kernel) {
  if (kernel.launch == nullptr) {
    HOST_LOG("Kernel callback is null");
    return false;
  }
  std::fill(problem.result.begin(), problem.result.end(), std::numeric_limits<float>::quiet_NaN());
  const KernelInputs inputs(problem, kernel.inputs);
  return on_device(
    inputs.a(), inputs.b(), problem.result,
    [&](const DeviceMatrix& a, const DeviceMatrix& b, DeviceMatrix& c) {
      return LAUNCH_CHECK(kernel.launch(a.data(), b.data(), c.data(), problem.shape, nullptr));
    }
  );
}

bool verify(const Problem& problem) {
  return lg::verify_values(
    problem.result, problem.expected, "matmul", kAbsoluteTolerance, kRelativeTolerance
  );
}

int check(Problem& problem, Kernel kernel) {
  return run_reference(problem) && run_kernel(problem, kernel) && verify(problem) ? 0 : 1;
}

} // namespace lg::matmul
