#include "checks.hpp"
#include "convolution/correctness.hpp"
#include "validation.hpp"

#include <algorithm>
#include <cstddef>
#include <cuda/buffer>
#include <nppi_filtering_functions.h>

namespace lg::convolution {
namespace {

using DeviceBuffer = cuda::device_buffer<float>;

bool make_npp_context(NppStreamContext& context) {
  context = {};
  int shared_per_block = 0;
  const bool ok =
    check_status(cudaGetDevice(&context.nCudaDeviceId)) &&
    check_status(device_attribute(cudaDevAttrMultiProcessorCount, context.nMultiProcessorCount)) &&
    check_status(
      device_attribute(cudaDevAttrMaxThreadsPerMultiProcessor, context.nMaxThreadsPerMultiProcessor)
    ) &&
    check_status(device_attribute(cudaDevAttrMaxThreadsPerBlock, context.nMaxThreadsPerBlock)) &&
    check_status(device_attribute(cudaDevAttrMaxSharedMemoryPerBlock, shared_per_block)) &&
    check_status(device_attribute(
      cudaDevAttrComputeCapabilityMajor, context.nCudaDevAttrComputeCapabilityMajor
    )) &&
    check_status(device_attribute(
      cudaDevAttrComputeCapabilityMinor, context.nCudaDevAttrComputeCapabilityMinor
    )) &&
    check_status(cudaStreamGetFlags(context.hStream, &context.nStreamFlags));
  context.nSharedMemPerBlock = static_cast<std::size_t>(shared_per_block);
  return ok;
}

bool run_reference(
  const Shape& shape, const DeviceBuffer& input, const DeviceBuffer& filter, DeviceBuffer& output
) {
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

  if (!check_status(cudaMemset2DAsync(padded.data(), padded_step, 0, padded_step, padded_height)) ||
      !check_status(cudaMemcpy2DAsync(
        padded.data() + source_offset, padded_step, input.data(), input_step, input_step, height,
        cudaMemcpyDeviceToDevice
      ))) {
    return false;
  }

  NppStreamContext context{};
  const auto* source = padded.data() + source_offset;
  // Preserve padding beyond the output ROI.
  // https://docs.nvidia.com/cuda/npp/image_filtering_functions.html
  return make_npp_context(context) &&
         check_status(nppiFilterBorder_32f_C1R_Ctx(
           source, static_cast<int>(padded_step), {padded_width, padded_height}, {radius, radius},
           output.data(), static_cast<int>(input_step), {width, height}, filter.data(),
           {filter_width, filter_width}, {radius, radius}, NPP_BORDER_REPLICATE, context
         ));
}

bool run_kernel(
  const Problem& problem,
  Kernel kernel,
  const DeviceBuffer& input,
  const DeviceBuffer& filter,
  DeviceBuffer& output
) {
  if (kernel.launch == nullptr) {
    HOST_LOG("Kernel callback is null");
    return false;
  }

  if (kernel.filter_setup != nullptr &&
      !check_status(kernel.filter_setup(filter.data(), problem.shape, nullptr))) {
    return false;
  }

  return LAUNCH_CHECK(
    kernel.launch(input.data(), filter.data(), output.data(), problem.shape, nullptr)
  );
}

bool verify(const Problem& problem, Kernel kernel) {
  return lg::verify_values(problem.result, problem.expected, kernel.name);
}

} // namespace

void fill_problem(Problem& problem) {
  validate_shape(problem.shape);
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
  std::ranges::fill(problem.expected, 0.0F);
  std::ranges::fill(problem.result, 0.0F);
}

int check(Problem& problem, Kernel kernel) {
  validate_shape(problem.shape);
  if (problem.input.size() != problem.shape.input_size() ||
      problem.filter.size() != problem.shape.filter_size()) {
    throw std::invalid_argument("convolution input size mismatch");
  }
  problem.expected.assign(problem.shape.input_size(), std::numeric_limits<float>::quiet_NaN());
  problem.result.assign(problem.shape.input_size(), std::numeric_limits<float>::quiet_NaN());
  const auto stream = default_stream();
  auto& pool = device_pool();
  // NPP expects reversed filter coefficients.
  auto npp_filter = problem.filter;
  std::ranges::reverse(npp_filter);
  const DeviceBuffer input{stream, pool, problem.input};
  const DeviceBuffer filter{stream, pool, problem.filter};
  const DeviceBuffer reference_filter{stream, pool, npp_filter};
  DeviceBuffer expected{stream, pool, problem.expected};
  DeviceBuffer result{stream, pool, problem.result};

  if (!run_reference(problem.shape, input, reference_filter, expected) ||
      !copy_checked(expected, problem.expected)) {
    return 1;
  }
  return run_kernel(problem, kernel, input, filter, result) &&
             copy_checked(result, problem.result) && verify(problem, kernel)
           ? 0
           : 1;
}

} // namespace lg::convolution
