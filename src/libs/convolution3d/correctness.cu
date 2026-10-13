#include "checks.hpp"
#include "convolution3d/correctness.hpp"
#include "validation.hpp"

#include <cuda/buffer>

namespace lg::convolution3d {
namespace {

using DeviceBuffer = cuda::device_buffer<float>;

std::size_t offset(Shape shape, int x, int y, int z) {
  return (static_cast<std::size_t>(z) * shape.height + y) * shape.width + x;
}

void run_reference(Problem& problem, std::size_t samples) {
  const int side = problem.shape.filter_width();
  const auto count = problem.shape.input_size();
  const auto checks = samples == 0 ? count : std::min(samples, count);
  for (std::size_t sample = 0; sample < checks; ++sample) {
    const auto index = lg::sample_index(sample, count, checks);
    const int x = static_cast<int>(index % problem.shape.width);
    const int y = static_cast<int>(index / problem.shape.width % problem.shape.height);
    const int z = static_cast<int>(index / problem.shape.width / problem.shape.height);
    float sum = 0.0F;
    for (int fz = 0; fz < side; ++fz) {
      const int iz = z + fz - problem.shape.radius;
      if (iz < 0 || iz >= problem.shape.depth) {
        continue;
      }
      for (int fy = 0; fy < side; ++fy) {
        const int iy = y + fy - problem.shape.radius;
        if (iy < 0 || iy >= problem.shape.height) {
          continue;
        }
        for (int fx = 0; fx < side; ++fx) {
          const int ix = x + fx - problem.shape.radius;
          if (ix >= 0 && ix < problem.shape.width) {
            const auto filter_index = (static_cast<std::size_t>(fz) * side + fy) * side + fx;
            sum += problem.input[offset(problem.shape, ix, iy, iz)] * problem.filter[filter_index];
          }
        }
      }
    }
    problem.expected[offset(problem.shape, x, y, z)] = sum;
  }
}

} // namespace

void fill_problem(Problem& problem) {
  validate_shape(problem.shape);
  problem.input.resize(problem.shape.input_size());
  problem.filter.resize(problem.shape.filter_size());
  problem.expected.assign(problem.shape.input_size(), 0.0F);
  problem.result.assign(problem.shape.input_size(), 0.0F);

  for (std::size_t index = 0; index < problem.input.size(); ++index) {
    problem.input[index] = static_cast<float>(index % 17) - 8.0F;
  }
  for (std::size_t index = 0; index < problem.filter.size(); ++index) {
    problem.filter[index] = static_cast<float>(index % 5) * 0.125F - 0.25F;
  }
}

int check(Problem& problem, Kernel kernel, std::size_t samples) {
  if (kernel.launch == nullptr) {
    HOST_LOG("Kernel callback is null");
    return 1;
  }

  validate_shape(problem.shape);
  if (problem.input.size() != problem.shape.input_size() ||
      problem.filter.size() != problem.shape.filter_size()) {
    throw std::invalid_argument("convolution input size mismatch");
  }
  problem.expected.assign(problem.shape.input_size(), std::numeric_limits<float>::quiet_NaN());
  problem.result.assign(problem.shape.input_size(), std::numeric_limits<float>::quiet_NaN());
  run_reference(problem, samples);

  const auto stream = default_stream();
  auto& pool = device_pool();
  const DeviceBuffer input{stream, pool, problem.input};
  const DeviceBuffer filter{stream, pool, problem.filter};
  DeviceBuffer result{stream, pool, problem.result};
  if (kernel.filter_setup != nullptr &&
      !CUDA_CHECK(kernel.filter_setup(filter.data(), problem.shape, nullptr))) {
    return 1;
  }

  return LAUNCH_CHECK(
           kernel.launch(input.data(), filter.data(), result.data(), problem.shape, nullptr)
         ) && COPY_CHECK(result, problem.result) &&
             lg::verify_sampled(problem.result, problem.expected, samples, kernel.name)
           ? 0
           : 1;
}

} // namespace lg::convolution3d
