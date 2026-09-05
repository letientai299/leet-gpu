#include "checks.hpp"

#include <cstddef>
#include <cuda/buffer>
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
  cuda::device_buffer<float> device_a{default_stream(), device_pool(), a.size(), cuda::no_init};
  cuda::device_buffer<float> device_b{default_stream(), device_pool(), b.size(), cuda::no_init};
  cuda::device_buffer<float> device_c{default_stream(), device_pool(), c.size(), cuda::no_init};

  if (!CUDA_CHECK(cudaMemcpy(device_a.data(), a.data(), bytes, cudaMemcpyHostToDevice)) ||
      !CUDA_CHECK(cudaMemcpy(device_b.data(), b.data(), bytes, cudaMemcpyHostToDevice)) ||
      !CUDA_CHECK(cudaMemset(device_c.data(), 0, bytes))) {
    return false;
  }

  constexpr unsigned threads = 256;
  const auto blocks = static_cast<unsigned>(cuda::ceil_div(a.size(), threads));
  vec_add_kernel<<<blocks, threads>>>(device_a.data(), device_b.data(), device_c.data(), a.size());

  return CUDA_CHECK(cudaGetLastError()) &&
         CUDA_CHECK(cudaMemcpy(c.data(), device_c.data(), bytes, cudaMemcpyDeviceToHost));
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
