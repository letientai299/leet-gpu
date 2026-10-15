#include "benchmark/profiler.hpp"
#include "checks.hpp"

#include <array>
#include <cstdio>
#include <cuda/buffer>
#include <cuda_fp16.h>
#include <mma.h>
#include <source_location>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr unsigned int kCount = 65537;
constexpr unsigned int kRepeats = 8;

void require(bool condition, std::source_location location = std::source_location::current()) {
  if (!condition) {
    throw std::runtime_error("Counter check failed at line " + std::to_string(location.line()));
  }
}

__global__ void fp32(float* values, unsigned int count) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  float value = values[index];
  for (unsigned int repeat = 0; repeat < kRepeats; ++repeat) {
    value = __fmaf_rn(value, 1.01F, 0.5F);
  }
  values[index] = value;
}

__global__ void integer(unsigned int* values, unsigned int count) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    values[index] += 1;
  }
}

__global__ void fp64(double* values, unsigned int count) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  double value = values[index];
  for (unsigned int repeat = 0; repeat < kRepeats; ++repeat) {
    value = __fma_rn(value, 1.01, 0.5);
  }
  values[index] = value;
}

__global__ void half_precision(__half2* values, unsigned int count) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    auto value = values[index];
    for (unsigned int repeat = 0; repeat < kRepeats; ++repeat) {
      value = __hfma2(value, __float2half2_rn(1.01F), __float2half2_rn(0.5F));
    }
    values[index] = value;
  }
}

__global__ void tensor(float* values) {
  using namespace nvcuda;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> result;
  wmma::fill_fragment(a, __float2half(0.5F));
  wmma::fill_fragment(b, __float2half(0.5F));
  wmma::fill_fragment(result, 0.0F);
  for (unsigned int repeat = 0; repeat < kRepeats; ++repeat) {
    wmma::mma_sync(result, a, b, result);
  }
  wmma::store_matrix_sync(
    values + static_cast<std::size_t>(blockIdx.x) * 256, result, 16, wmma::mem_row_major
  );
}

__global__ void read_input(const unsigned int* values, unsigned int count, unsigned int* output) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    atomicAdd(&output[values[index] % 7], 1);
  }
}

__global__ void atomics(unsigned int* values, unsigned int count) {
  __shared__ unsigned int shared;
  if (threadIdx.x == 0) {
    shared = 0;
  }
  __syncthreads();
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    const auto previous = atomicAdd(&values[index], 1);
    values[index] = previous + 2;
    atomicAdd(&values[count], 1);
    atomicAdd(&shared, 1);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    values[count + 1 + blockIdx.x] = shared;
  }
}

void check_counters() {
  int device = 0;
  require(check_status(cudaGetDevice(&device)));
  const auto stream = default_stream();
  auto& pool = device_pool();
  const std::vector<float> host(kCount, 0.5F);
  cuda::device_buffer<float> values{stream, pool, host};
  require(check_status(cudaDeviceSynchronize()));
  const auto measured = lg::benchmark::profile_hardware(device, nullptr, [&](cudaStream_t timed) {
    fp32<<<(kCount + 127) / 128, 128, 0, timed>>>(values.data(), kCount);
  });
  if (!measured) {
    throw std::runtime_error("FP32 counters unavailable");
  }
  require(measured->fp32_flops == static_cast<double>(kCount) * kRepeats * 2);
  require(measured->fp32_flops_per_second > 0 && measured->fp32_peak > 0);
  require(measured->dram_read_bytes > 0 && measured->dram_bytes_per_second > 0);
  require(measured->occupancy > 0 && measured->occupancy <= 1);
  std::printf("Measured FP32 FLOPs: %.0f\n", measured->fp32_flops);
  std::printf("Measured DRAM reads: %.0f bytes\n", measured->dram_read_bytes);
}

void check_integer(int device) {
  const auto stream = default_stream();
  auto& pool = device_pool();
  const std::vector<unsigned int> host_integers(2 * static_cast<std::size_t>(kCount), 0);
  cuda::device_buffer<unsigned int> integers{stream, pool, host_integers};
  require(check_status(cudaDeviceSynchronize()));
  const auto integer_metrics =
    lg::benchmark::profile_hardware(device, nullptr, [&](cudaStream_t timed) {
      integer<<<(kCount + 127) / 128, 128, 0, timed>>>(integers.data(), kCount);
    });
  if (!integer_metrics) {
    throw std::runtime_error("Integer counters unavailable");
  }
  require(integer_metrics->fp32_flops == 0);
  require(integer_metrics->dram_read_bytes > 0);
  require(integer_metrics->integer_instructions > 0);
  require(integer_metrics->integer_instructions_per_second > 0);
  require(integer_metrics->integer_utilization > 0);
  require(integer_metrics->atomic_sectors == 0 && integer_metrics->reduction_sectors == 0);
  require(integer_metrics->shared_atomics == 0);

  const auto atomic_metrics =
    lg::benchmark::profile_hardware(device, nullptr, [&](cudaStream_t timed) {
      atomics<<<(kCount + 127) / 128, 128, 0, timed>>>(integers.data(), kCount);
    });
  if (!atomic_metrics) {
    throw std::runtime_error("Atomic counters unavailable");
  }
  require(atomic_metrics->global_atomics > 0 && atomic_metrics->global_reductions > 0);
  require(atomic_metrics->shared_atomics > 0 && atomic_metrics->shared_atomic_utilization > 0);
  require(atomic_metrics->atomic_sectors > 0 && atomic_metrics->atomic_sectors_per_second > 0);
  require(atomic_metrics->atomic_utilization > 0);
  require(
    atomic_metrics->reduction_sectors > 0 && atomic_metrics->reduction_sectors_per_second > 0
  );
  require(atomic_metrics->reduction_utilization > 0);
}

void check_double(int device) {
  const auto stream = default_stream();
  auto& pool = device_pool();
  const std::vector<double> host_doubles(kCount, 0.5);
  cuda::device_buffer<double> doubles{stream, pool, host_doubles};
  require(check_status(cudaDeviceSynchronize()));
  const auto double_metrics =
    lg::benchmark::profile_hardware(device, nullptr, [&](cudaStream_t timed) {
      fp64<<<(kCount + 127) / 128, 128, 0, timed>>>(doubles.data(), kCount);
    });
  if (!double_metrics) {
    throw std::runtime_error("FP64 counters unavailable");
  }
  require(double_metrics->fp64_flops == static_cast<double>(kCount) * kRepeats * 2);
  require(double_metrics->fp64_flops_per_second > 0 && double_metrics->fp64_peak > 0);
}

void check_half(int device) {
  const auto stream = default_stream();
  const std::vector<__half2> host(kCount, __float2half2_rn(0.5F));
  cuda::device_buffer<__half2> values{stream, device_pool(), host};
  require(check_status(cudaDeviceSynchronize()));
  const auto metrics = lg::benchmark::profile_hardware(device, nullptr, [&](cudaStream_t timed) {
    half_precision<<<(kCount + 127) / 128, 128, 0, timed>>>(values.data(), kCount);
  });
  if (!metrics) {
    throw std::runtime_error("FP16 counters unavailable");
  }
  require(metrics->fp16_utilization > 0);
}

void check_tensor(int device) {
  const auto stream = default_stream();
  const std::vector<float> host(32768, 0.0F);
  cuda::device_buffer<float> values{stream, device_pool(), host};
  require(check_status(cudaDeviceSynchronize()));
  const auto metrics = lg::benchmark::profile_hardware(device, nullptr, [&](cudaStream_t timed) {
    tensor<<<128, 32, 0, timed>>>(values.data());
  });
  if (!metrics) {
    throw std::runtime_error("Tensor counters unavailable");
  }
  require(metrics->tensor_utilization > 0);
}

void check_read_traffic(int device) {
  constexpr unsigned int count = 6 * 1024 * 1024;
  const auto stream = default_stream();
  auto& pool = device_pool();
  const std::vector<unsigned int> host(count, 1);
  cuda::device_buffer<unsigned int> values{stream, pool, host};
  const std::array<unsigned int, 7> zeros{};
  cuda::device_buffer<unsigned int> output{stream, pool, zeros};
  require(check_status(cudaDeviceSynchronize()));
  const auto metrics = lg::benchmark::profile_hardware(device, nullptr, [&](cudaStream_t timed) {
    require(check_status(cudaMemsetAsync(output.data(), 0, sizeof(zeros), timed)));
    read_input<<<(count + 127) / 128, 128, 0, timed>>>(values.data(), count, output.data());
  });
  if (!metrics) {
    throw std::runtime_error("DRAM counters unavailable");
  }
  std::printf(
    "Read-only input DRAM reads/writes: %.0f / %.0f bytes\n", metrics->dram_read_bytes,
    metrics->dram_write_bytes
  );
  require(
    metrics->dram_read_bytes >= 0.9 * static_cast<double>(host.size() * sizeof(unsigned int))
  );
  require(metrics->dram_write_bytes < 1024 * 1024);
  std::printf("Read-only input DRAM writes: %.0f bytes\n", metrics->dram_write_bytes);
}

} // namespace

int main() {
  return guarded([] {
    if (!init_host()) {
      return 1;
    }
    check_counters();
    check_integer(0);
    check_double(0);
    check_half(0);
    check_tensor(0);
    check_read_traffic(0);
    return 0;
  });
}
