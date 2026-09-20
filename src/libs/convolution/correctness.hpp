#pragma once

#include "convolution/kernel.hpp"

#include <vector>

namespace lg::convolution {

struct Problem {
  Shape shape;
  std::vector<float> input;
  std::vector<float> filter;
  std::vector<float> expected;
  std::vector<float> result;
};

void fill_problem(Problem& problem);
int check(Problem& problem, Kernel kernel);

} // namespace lg::convolution
