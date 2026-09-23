#include "convolution/kernel.hpp"
#include "convolution3d/kernel.hpp"
#include "matmul/matrix.hpp"
#include "validation.hpp"

#include <climits>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition) {
  if (!condition) {
    throw std::runtime_error("host regression failed");
  }
}

template <typename Function> void rejects(Function function) {
  bool rejected = false;
  try {
    function();
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected);
}

} // namespace

int main() try {
  const auto max = std::numeric_limits<std::size_t>::max();
  require(!lg::checked_mul(max, 2));
  require(lg::checked_mul(max, 0) == 0);
  require(!lg::checked_add(max, 1));
  require(lg::checked_add(max, 0) == max);
  for (int length = 1; length < 40; ++length) {
    for (int radius = 0; radius < 50; ++radius) {
      std::size_t pairs = 0;
      for (int x = 0; x < length; ++x) {
        for (int y = 0; y < length; ++y) {
          pairs += std::abs(x - y) <= radius ? 1 : 0;
        }
      }
      require(lg::valid_axis_pairs(length, radius) == pairs);
    }
  }
  rejects([] {
    lg::convolution::validate_shape({0, 1, 0});
  });
  rejects([] {
    lg::convolution::validate_shape({1, 1, -1});
  });
  rejects([] {
    lg::convolution::validate_shape({INT_MAX, 1, 1});
  });
  rejects([] {
    lg::convolution3d::validate_shape({INT_MAX, INT_MAX, INT_MAX, 0});
  });
  lg::convolution::validate_shape({1, 1, 4});
  lg::convolution3d::validate_shape({1, 2, 3, 4});

  lg::matmul::Matrix matrix(2, 3);
  matrix(1, 2) = 7;
  require(matrix(1, 2) == 7);
  rejects([&] {
    matrix(0, 3) = 1;
  });
  const auto& constant = matrix;
  rejects([&] {
    (void)constant(0, 3);
  });
  rejects([&] {
    lg::matmul::fill_random(matrix, {1, 2, 1});
  });
  rejects([&] {
    lg::matmul::fill_random(matrix, {1, NAN, 1});
  });

  const std::vector<float> zeros(4, 0);
  require(lg::verify_values(zeros, zeros, "zeros"));
  auto actual = zeros;
  actual[2] = NAN;
  require(!lg::verify_values(actual, zeros, "unwritten"));
  require(!lg::verify_values(zeros, actual, "invalid reference"));
  actual[2] = INFINITY;
  require(!lg::verify_values(zeros, actual, "infinite reference"));
  actual[2] = 1;
  require(!lg::verify_values(actual, zeros, "mismatch"));
  actual.pop_back();
  require(!lg::verify_values(actual, zeros, "size"));
  lg::ErrorStats stats;
  stats.add(1, 0, 1.0e-5, 1.0e-4);
  stats.add(3, 1, 1.0e-5, 1.0e-4);
  require(stats.mismatches == 2 && stats.max_absolute == 2 && stats.squared_error == 5);
  std::puts("Host regressions passed");
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
