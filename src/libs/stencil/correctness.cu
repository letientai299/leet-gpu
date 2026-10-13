#include "checks.hpp"
#include "stencil/correctness.hpp"
#include "validation.hpp"

#include <algorithm>
#include <cstdint>
#include <cuda/buffer>
#include <limits>

namespace lg::stencil {
namespace {

using DeviceBuffer = cuda::device_buffer<float>;

std::size_t offset(Shape shape, int x, int y, int z) {
  return (static_cast<std::size_t>(z) * shape.height + y) * shape.width + x;
}

bool is_interior(Shape shape, int x, int y, int z) {
  return x > 0 && x < shape.width - 1 && y > 0 && y < shape.height - 1 && z > 0 &&
         z < shape.depth - 1;
}

float reference_at(const Problem& problem, std::size_t index) {
  const Shape s = problem.shape;
  const auto& in = problem.input;
  const int x = static_cast<int>(index % s.width);
  const int y = static_cast<int>(index / s.width % s.height);
  const int z = static_cast<int>(index / s.plane_size());
  if (!is_interior(s, x, y, z)) {
    return in[index];
  }
  const Coefficients c = problem.coefficients;
  return c.c0 * in[offset(s, x, y, z)] + c.c1 * in[offset(s, x - 1, y, z)] +
         c.c2 * in[offset(s, x + 1, y, z)] + c.c3 * in[offset(s, x, y - 1, z)] +
         c.c4 * in[offset(s, x, y + 1, z)] + c.c5 * in[offset(s, x, y, z - 1)] +
         c.c6 * in[offset(s, x, y, z + 1)];
}

// Fills only the sampled positions unless samples is 0.
void run_reference(Problem& problem, std::size_t samples) {
  const auto count = problem.shape.grid_size();
  const auto checks = samples == 0 ? count : std::min(samples, count);
  problem.expected.assign(count, std::numeric_limits<float>::quiet_NaN());
  for (std::size_t sample = 0; sample < checks; ++sample) {
    const auto index = lg::sample_index(sample, count, checks);
    problem.expected[index] = reference_at(problem, index);
  }
}

// lowbias32 hash: neighbors at any stride get unrelated values.
// https://github.com/skeeto/hash-prospector
float hashed_value(std::size_t index) {
  auto x = static_cast<std::uint32_t>(index);
  x ^= x >> 16U;
  x *= 0x7feb352dU;
  x ^= x >> 15U;
  x *= 0x846ca68bU;
  x ^= x >> 16U;
  // Small integers keep every weighted sum exact in float.
  return static_cast<float>(x >> 27U) - 16.0F;
}

} // namespace

void fill_problem(Problem& problem) {
  validate_shape(problem.shape);
  // Distinct weights and hashed inputs expose swapped axes or neighbors.
  problem.coefficients = {0.25F, 0.0625F, 0.125F, 0.1875F, 0.3125F, 0.375F, 0.4375F};
  problem.input.resize(problem.shape.grid_size());
  for (std::size_t index = 0; index < problem.input.size(); ++index) {
    problem.input[index] = hashed_value(index);
  }
}

std::vector<float> initial_output(const Problem& problem) {
  const Shape s = problem.shape;
  std::vector<float> output = problem.input;
  for (int z = 0; z < s.depth; ++z) {
    for (int y = 0; y < s.height; ++y) {
      for (int x = 0; x < s.width; ++x) {
        if (is_interior(s, x, y, z)) {
          output[offset(s, x, y, z)] = std::numeric_limits<float>::quiet_NaN();
        }
      }
    }
  }
  return output;
}

int check(Problem& problem, Kernel kernel, std::size_t samples) {
  if (kernel.launch == nullptr) {
    HOST_LOG("Kernel callback is null");
    return 1;
  }

  validate_shape(problem.shape);
  if (problem.input.size() != problem.shape.grid_size()) {
    throw std::invalid_argument("stencil input size mismatch");
  }
  run_reference(problem, samples);
  problem.result = initial_output(problem);

  const auto stream = default_stream();
  auto& pool = device_pool();
  const DeviceBuffer input{stream, pool, problem.input};
  DeviceBuffer result{stream, pool, problem.result};

  return LAUNCH_CHECK(
           kernel.launch(input.data(), result.data(), problem.shape, problem.coefficients, nullptr)
         ) && COPY_CHECK(result, problem.result) &&
             lg::verify_sampled(problem.result, problem.expected, samples, kernel.name)
           ? 0
           : 1;
}

} // namespace lg::stencil
