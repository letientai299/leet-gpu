#pragma once

#include "stencil/kernel.hpp"

#include <vector>

namespace lg::stencil {

struct Problem {
  Shape shape;
  Coefficients coefficients;
  std::vector<float> input;
  std::vector<float> expected;
  std::vector<float> result;
};

void fill_problem(Problem& problem);
// Output starts as input with a NaN interior. Kernels write only the interior.
std::vector<float> initial_output(const Problem& problem);
// Compares every value when samples is 0, else that many spread positions.
int check(Problem& problem, Kernel kernel, std::size_t samples = 0);

} // namespace lg::stencil
