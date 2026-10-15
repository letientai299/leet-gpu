#pragma once

#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace lg::reduction {

struct Problem {
  std::vector<float> input;
  float expected;
};

inline Problem make_problem(unsigned int length, unsigned int segment_size, unsigned int pattern) {
  if (segment_size == 0) {
    throw std::invalid_argument("reduction segment size is zero");
  }
  const auto segments = length == 0 ? 1 : (static_cast<std::size_t>(length) - 1) / segment_size + 1;
  Problem problem{std::vector<float>(segments * segment_size, 0.0F), 0.0F};
  for (unsigned int i = 0; i < length; ++i) {
    switch (pattern) {
    case 0:
      problem.input[i] = 0.0F;
      break;
    case 1:
      problem.input[i] = 1.0F;
      break;
    case 2:
      problem.input[i] = -0.25F;
      break;
    case 3:
      problem.input[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.125F;
      break;
    case 4:
      problem.input[i] = i == length - 1 ? 7.0F : 0.0F;
      break;
    default:
      throw std::invalid_argument("reduction input pattern is invalid");
    }
  }
  // Dyadic inputs keep these test sums exact.
  problem.expected =
    static_cast<float>(std::accumulate(problem.input.begin(), problem.input.end(), 0.0));
  return problem;
}

} // namespace lg::reduction
