#include "benchmark/captured.hpp"
#include "checks.hpp"
#include "log.hpp"
#include "matmul/benchmark.hpp"
#include "matmul/kernel.hpp"

#include <cstddef>
#include <cuda/buffer>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {

namespace mm = lg::matmul;

constexpr unsigned kBenchSize = 4096;
// cuBLAS recommends 32 MiB on sm_90+.
// https://docs.nvidia.com/cuda/cublas/#cublassetworkspace
constexpr std::size_t kWorkspaceBytes = std::size_t{32} << 20;

struct Cublas {
  CublasHandle handle;
  // A fixed workspace keeps stream capture free of allocations.
  cuda::device_buffer<std::byte> workspace{default_stream(), device_pool(), kWorkspaceBytes,
                                           cuda::no_init};
  std::optional<cudaStream_t> stream;
};

/// Returns the current device's cuBLAS state, created on first use.
Cublas& get_cublas() {
  // A handle binds to one device, and NVBench may run on several.
  static std::vector<std::unique_ptr<Cublas>> by_device = [] {
    int count = 0;
    if (!CUDA_CHECK(cudaGetDeviceCount(&count))) {
      throw std::runtime_error("CUDA device count failed");
    }
    return std::vector<std::unique_ptr<Cublas>>(static_cast<std::size_t>(count));
  }();
  int device = 0;
  if (!CUDA_CHECK(cudaGetDevice(&device))) {
    throw std::runtime_error("CUDA device query failed");
  }
  auto& slot = by_device.at(static_cast<std::size_t>(device));
  if (!slot) {
    slot = std::make_unique<Cublas>();
    // The host's shared cuBLAS may differ from the build toolkit.
    int version = 0;
    if (CUBLAS_CHECK(cublasGetVersion(slot->handle.get(), &version))) {
      HOST_LOG("cuBLAS {} on device {}", version, device);
    }
  }
  return *slot;
}

void launch_matmul_cublas(
  const float* a, const float* b, float* c, const mm::GemmShape& shape, cudaStream_t stream
) {
  auto& blas = get_cublas();
  auto* const handle = blas.handle.get();
  if (blas.stream != stream) {
    // cublasSetStream drops the user workspace, so attach it again.
    if (!CUBLAS_CHECK(cublasSetStream(handle, stream)) ||
        !CUBLAS_CHECK(cublasSetWorkspace(handle, blas.workspace.data(), blas.workspace.size()))) {
      throw std::runtime_error("cuBLAS stream setup failed");
    }
    blas.stream = stream;
  }
  const auto m = static_cast<int>(shape.m());
  const auto n = static_cast<int>(shape.n());
  const auto k = static_cast<int>(shape.k());
  const float alpha = 1;
  const float beta = 0;
  // cuBLAS is column-major: row-major C = A B is C^T = B^T A^T, so swap A and B.
  // Throw so NVBench fails the state instead of timing nothing.
  if (!CUBLAS_CHECK(
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, n, a, k, &beta, c, n)
      )) {
    throw std::runtime_error("cuBLAS Sgemm failed");
  }
}

cudaError_t
cublas_resources(const mm::GemmShape& shape, lg::benchmark::KernelResources& resources) {
  // Create the handle before capture; creation allocates.
  get_cublas();
  // Real buffers keep cuBLAS's alignment-based kernel choice.
  auto& pool = device_pool();
  const cuda::device_buffer<float> a{default_stream(), pool, shape.a_size(), cuda::no_init};
  const cuda::device_buffer<float> b{default_stream(), pool, shape.b_size(), cuda::no_init};
  cuda::device_buffer<float> c{default_stream(), pool, shape.c_size(), cuda::no_init};
  return lg::benchmark::inspect_launch(
    [&](cudaStream_t stream) {
      launch_matmul_cublas(a.data(), b.data(), c.data(), shape, stream);
    },
    resources
  );
}

const mm::Kernel kMatmulCublas{
  "matmul.cublas", launch_matmul_cublas, nullptr, {}, cublas_resources};

} // namespace

int main(int argc, char** argv) {
  return mm::run_app(
    argc, argv, {kMatmulCublas, std::nullopt, mm::GemmShape(kBenchSize, kBenchSize, kBenchSize)}
  );
}
