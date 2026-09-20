#include "checks.hpp"
#include "convolution/correctness.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cuda/buffer>
#include <nppi_filtering_functions.h>

namespace lg::convolution {
namespace {

using DeviceBuffer = cuda::device_buffer<float>;

bool make_npp_context(NppStreamContext& context) {
  constexpr cudaStream_t stream = nullptr;
  int device = 0;
  int major = 0;
  int minor = 0;
  unsigned int stream_flags = 0;
  cudaDeviceProp properties{};
  if (!CUDA_CHECK(cudaGetDevice(&device)) ||
      !CUDA_CHECK(cudaGetDeviceProperties(&properties, device)) ||
      !CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device)) ||
      !CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device)) ||
      !CUDA_CHECK(cudaStreamGetFlags(stream, &stream_flags))) {
    return false;
  }

  context = {};
  context.hStream = stream;
  context.nCudaDeviceId = device;
  context.nMultiProcessorCount = properties.multiProcessorCount;
  context.nMaxThreadsPerMultiProcessor = properties.maxThreadsPerMultiProcessor;
  context.nMaxThreadsPerBlock = properties.maxThreadsPerBlock;
  context.nSharedMemPerBlock = properties.sharedMemPerBlock;
  context.nCudaDevAttrComputeCapabilityMajor = major;
  context.nCudaDevAttrComputeCapabilityMinor = minor;
  context.nStreamFlags = stream_flags;
  return true;
}

bool run_reference(const Shape& shape,
                   const DeviceBuffer& input,
                   const DeviceBuffer& filter,
                   DeviceBuffer& output) {
  const int width = shape.width;
  const int height = shape.height;
  const int radius = shape.radius;
  const int filter_width = shape.filter_width();
  const int padded_width = width + 2 * radius;
  const int padded_height = height + 2 * radius;
  const std::size_t input_step = static_cast<std::size_t>(width) * sizeof(float);
  const std::size_t padded_step = static_cast<std::size_t>(padded_width) * sizeof(float);
  const std::size_t source_offset = static_cast<std::size_t>(radius) * padded_width + radius;
  DeviceBuffer padded{default_stream(), device_pool(),
                      static_cast<std::size_t>(padded_width) * padded_height, cuda::no_init};

  if (!CUDA_CHECK(cudaMemset2DAsync(padded.data(), padded_step, 0, padded_step, padded_height)) ||
      !CUDA_CHECK(cudaMemcpy2DAsync(padded.data() + source_offset, padded_step, input.data(),
                                    input_step, input_step, height, cudaMemcpyDeviceToDevice))) {
    return false;
  }

  NppStreamContext context{};
  const auto* source = padded.data() + source_offset;
  return make_npp_context(context) &&
         NPP_CHECK(nppiFilter_32f_C1R_Ctx(source, static_cast<int>(padded_step), output.data(),
                                          static_cast<int>(input_step), {width, height},
                                          filter.data(), {filter_width, filter_width},
                                          {radius, radius}, context));
}

bool run_kernel(const Problem& problem,
                Kernel kernel,
                const DeviceBuffer& input,
                const DeviceBuffer& filter,
                DeviceBuffer& output) {
  if (kernel.launch == nullptr) {
    HOST_LOG("Kernel callback is null");
    return false;
  }

  cudaGetLastError();
  kernel.launch(input.data(), filter.data(), output.data(), problem.shape, nullptr);
  return CUDA_CHECK(cudaGetLastError());
}

bool verify(const Problem& problem, Kernel kernel) {
  constexpr float absolute_tolerance = 1.0e-5F;
  constexpr float relative_tolerance = 1.0e-4F;
  for (std::size_t index = 0; index < problem.result.size(); ++index) {
    const float actual = problem.result[index];
    const float expected = problem.expected[index];
    const float difference = std::fabs(actual - expected);
    const float tolerance = absolute_tolerance + relative_tolerance * std::fabs(expected);
    if (!std::isfinite(actual) || difference > tolerance) {
      HOST_LOG("%s mismatch at %zu: %g vs %g", kernel.name, index, actual, expected);
      return false;
    }
  }
  HOST_LOG("%s passed: %zu values", kernel.name, problem.result.size());
  return true;
}

} // namespace

void fill_problem(Problem& problem) {
  problem.input.resize(problem.shape.input_size());
  problem.filter.resize(problem.shape.filter_size());
  problem.expected.resize(problem.shape.input_size());
  problem.result.resize(problem.shape.input_size());

  for (std::size_t index = 0; index < problem.input.size(); ++index) {
    problem.input[index] = static_cast<float>(index % 17) - 8.0F;
  }
  for (std::size_t index = 0; index < problem.filter.size(); ++index) {
    problem.filter[index] = static_cast<float>(index % 5) * 0.125F - 0.25F;
  }
  std::fill(problem.expected.begin(), problem.expected.end(), 0.0F);
  std::fill(problem.result.begin(), problem.result.end(), 0.0F);
}

int check(Problem& problem, Kernel kernel) {
  const auto stream = default_stream();
  auto& pool = device_pool();
  // NPP expects reversed filter coefficients.
  auto npp_filter = problem.filter;
  std::reverse(npp_filter.begin(), npp_filter.end());
  const DeviceBuffer input{stream, pool, problem.input};
  const DeviceBuffer filter{stream, pool, problem.filter};
  const DeviceBuffer reference_filter{stream, pool, npp_filter};
  DeviceBuffer expected{stream, pool, problem.expected};
  DeviceBuffer result{stream, pool, problem.result};

  if (!run_reference(problem.shape, input, reference_filter, expected) ||
      !COPY_CHECK(expected, problem.expected)) {
    return 1;
  }
  return run_kernel(problem, kernel, input, filter, result) && COPY_CHECK(result, problem.result) &&
                 verify(problem, kernel)
             ? 0
             : 1;
}

} // namespace lg::convolution
