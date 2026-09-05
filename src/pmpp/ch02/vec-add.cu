#include "checks.hpp"

#include <cstddef>
#include <vector>

namespace {

// RAII: destructor frees owned device buffers.
class DeviceVectors {
public:
  DeviceVectors() = default;
  DeviceVectors(const DeviceVectors&) = delete;
  DeviceVectors& operator=(const DeviceVectors&) = delete;

  ~DeviceVectors() {
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
  }

  bool allocate(std::size_t bytes) {
    return CUDA_CHECK(cudaMalloc(&a, bytes)) && CUDA_CHECK(cudaMalloc(&b, bytes)) &&
           CUDA_CHECK(cudaMalloc(&c, bytes));
  }

  float* a = nullptr;
  float* b = nullptr;
  float* c = nullptr;
};

__global__ void vec_add_kernel(const float* a, const float* b, float* c, std::size_t count) {
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    c[idx] = a[idx] + b[idx];
  }
}

void init_vectors(std::vector<float>& a, std::vector<float>& b, std::vector<float>& expected) {
  for (std::size_t index = 0; index < a.size(); ++index) {
    a[index] = static_cast<float>(index) * 0.5F;
    b[index] = static_cast<float>(index % 17) * 1.25F;
    expected[index] = a[index] + b[index];
  }
}

bool add_on_gpu(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c) {
  const std::size_t bytes = a.size() * sizeof(float);
  DeviceVectors device;
  if (!device.allocate(bytes)) {
    return false;
  }

  if (!CUDA_CHECK(cudaMemcpy(device.a, a.data(), bytes, cudaMemcpyHostToDevice)) ||
      !CUDA_CHECK(cudaMemcpy(device.b, b.data(), bytes, cudaMemcpyHostToDevice)) ||
      !CUDA_CHECK(cudaMemset(device.c, 0, bytes))) {
    return false;
  }

  constexpr unsigned threads = 256;
  const auto blocks = static_cast<unsigned>((a.size() + threads - 1) / threads);
  vec_add_kernel<<<blocks, threads>>>(device.a, device.b, device.c, a.size());

  return CUDA_CHECK(cudaGetLastError()) &&
         CUDA_CHECK(cudaMemcpy(c.data(), device.c, bytes, cudaMemcpyDeviceToHost));
}

bool verify_output(const std::vector<float>& c, const std::vector<float>& expected) {
  for (std::size_t index = 0; index < c.size(); ++index) {
    if (c[index] != expected[index]) {
      HOST_LOG("Mismatch at %zu: %.1f != %.1f", index, c[index], expected[index]);
      return false;
    }
  }
  HOST_LOG("Vector addition passed: %zu values", c.size());
  return true;
}

bool run_vec_add() {
  constexpr std::size_t count = 600;
  std::vector<float> a(count);
  std::vector<float> b(count);
  std::vector<float> expected(count);
  std::vector<float> c(count);
  init_vectors(a, b, expected);

  return add_on_gpu(a, b, c) && verify_output(c, expected);
}

} // namespace

int main() {
  init_log();
  if (!init_cuda()) {
    return 1;
  }
  return run_vec_add() ? 0 : 1;
}
