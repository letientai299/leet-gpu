#include "device.hpp"

#include <cstddef>
#include <cuda/cmath>
#include <vector>

namespace {

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
  DeviceBuffer<float> device_a;
  DeviceBuffer<float> device_b;
  DeviceBuffer<float> device_c;
  if (!device_a.allocate(bytes) || !device_b.allocate(bytes) || !device_c.allocate(bytes)) {
    return false;
  }

  if (!CUDA_CHECK(cudaMemcpy(device_a.get(), a.data(), bytes, cudaMemcpyHostToDevice)) ||
      !CUDA_CHECK(cudaMemcpy(device_b.get(), b.data(), bytes, cudaMemcpyHostToDevice)) ||
      !CUDA_CHECK(cudaMemset(device_c.get(), 0, bytes))) {
    return false;
  }

  constexpr unsigned threads = 256;
  const auto blocks = static_cast<unsigned>(cuda::ceil_div(a.size(), threads));
  vec_add_kernel<<<blocks, threads>>>(device_a.get(), device_b.get(), device_c.get(), a.size());

  return CUDA_CHECK(cudaGetLastError()) &&
         CUDA_CHECK(cudaMemcpy(c.data(), device_c.get(), bytes, cudaMemcpyDeviceToHost));
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

int run_vec_add() {
  constexpr std::size_t count = 600;
  std::vector<float> a(count);
  std::vector<float> b(count);
  std::vector<float> expected(count);
  std::vector<float> c(count);
  init_vectors(a, b, expected);
  return add_on_gpu(a, b, c) && verify_output(c, expected) ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
  return run_host(argc, argv, run_vec_add);
}
