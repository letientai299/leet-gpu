#pragma once

#include "convolution3d/kernel.hpp"

#include <vector>

namespace lg::convolution3d {

struct Problem {
  Shape shape;
  std::vector<float> input;
  std::vector<float> filter;
  std::vector<float> expected;
  std::vector<float> result;
};

void fill_problem(Problem& problem);
int check(Problem& problem, Kernel kernel, std::size_t samples = 0);

} // namespace lg::convolution3d
