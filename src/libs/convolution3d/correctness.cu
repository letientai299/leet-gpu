#include "checks.hpp"
#include "convolution3d/correctness.hpp"

#include <cmath>
#include <cuda/buffer>

namespace lg::convolution3d {
namespace {

using DeviceBuffer = cuda::device_buffer<float>;

std::size_t offset(Shape shape, int x, int y, int z) {
  return (static_cast<std::size_t>(z) * shape.height + y) * shape.width + x;
}

void run_reference(Problem& problem) {
  const int side = problem.shape.filter_width();
  for (int z = 0; z < problem.shape.depth; ++z) {
    for (int y = 0; y < problem.shape.height; ++y) {
      for (int x = 0; x < problem.shape.width; ++x) {
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
                sum +=
                  problem.input[offset(problem.shape, ix, iy, iz)] * problem.filter[filter_index];
              }
            }
          }
        }
        problem.expected[offset(problem.shape, x, y, z)] = sum;
      }
    }
  }
}

bool verify(const Problem& problem, Kernel kernel) {
  constexpr float kAbsoluteTolerance = 1.0e-5F;
  constexpr float kRelativeTolerance = 1.0e-4F;
  for (std::size_t index = 0; index < problem.result.size(); ++index) {
    const float actual = problem.result[index];
    const float expected = problem.expected[index];
    const float difference = std::fabs(actual - expected);
    const float tolerance = kAbsoluteTolerance + kRelativeTolerance * std::fabs(expected);
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
  problem.expected.assign(problem.shape.input_size(), 0.0F);
  problem.result.assign(problem.shape.input_size(), 0.0F);

  for (std::size_t index = 0; index < problem.input.size(); ++index) {
    problem.input[index] = static_cast<float>(index % 17) - 8.0F;
  }
  for (std::size_t index = 0; index < problem.filter.size(); ++index) {
    problem.filter[index] = static_cast<float>(index % 5) * 0.125F - 0.25F;
  }
  run_reference(problem);
}

int check(Problem& problem, Kernel kernel) {
  if (kernel.launch == nullptr) {
    HOST_LOG("Kernel callback is null");
    return 1;
  }

  const auto stream = default_stream();
  auto& pool = device_pool();
  const DeviceBuffer input{stream, pool, problem.input};
  const DeviceBuffer filter{stream, pool, problem.filter};
  DeviceBuffer result{stream, pool, problem.result};
  if (kernel.filter_setup != nullptr &&
      !CUDA_CHECK(kernel.filter_setup(filter.data(), problem.shape, nullptr))) {
    return 1;
  }

  cudaGetLastError();
  kernel.launch(input.data(), filter.data(), result.data(), problem.shape, nullptr);
  return CUDA_CHECK(cudaGetLastError()) && COPY_CHECK(result, problem.result) &&
             verify(problem, kernel)
           ? 0
           : 1;
}

} // namespace lg::convolution3d
