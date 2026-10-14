#include "benchmark/workload.hpp"
#include "convolution/kernel.hpp"
#include "convolution3d/kernel.hpp"
#include "matmul/matrix.hpp"
#include "validation.hpp"

#include <climits>
#include <cstdio>
#include <limits>
#include <optional>
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

constexpr auto kMaxSize = std::numeric_limits<std::size_t>::max();

void check_arithmetic() {
  static_assert(!lg::checked_mul(kMaxSize, 2));
  static_assert(lg::checked_mul(kMaxSize, 0) == 0);
  static_assert(!lg::checked_add(kMaxSize, 1));
  static_assert(lg::checked_add(kMaxSize, 0) == kMaxSize);
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
}

void check_shapes() {
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
}

void check_matrix() {
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
}

void check_verify() {
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
  const std::vector<float> longer(8, 0);
  require(!lg::verify_sampled(longer, zeros, 2, "sampled size"));
  require(lg::verify_sampled(longer, longer, 3, "sampled"));
}

void check_workload() {
  const auto workload = lg::benchmark::float_workload({}, 1, 1, 1);
  require(workload && workload->buffer_bytes == 8 && !workload->no_reuse);
  require(!lg::benchmark::float_workload({}, 1, kMaxSize, 1));
  require(!lg::benchmark::float_workload({}, std::nullopt, 1, 1));
  const auto overflow = +[](int, lg::benchmark::Traffic& traffic) {
    traffic = {kMaxSize, 0};
    return true;
  };
  require(!lg::benchmark::with_no_reuse(workload, overflow, 0));
  const auto doubled = +[](int, lg::benchmark::Traffic& traffic) {
    traffic = {2, 1};
    return true;
  };
  const auto modeled = lg::benchmark::with_no_reuse(workload, doubled, 0);
  require(modeled && modeled->no_reuse && modeled->no_reuse->read_bytes == 8);
}

void check_error_stats() {
  lg::ErrorStats stats;
  stats.add(1, 0, 1.0e-5, 1.0e-4);
  stats.add(3, 1, 1.0e-5, 1.0e-4);
  require(stats.mismatches == 2 && stats.max_absolute == 2 && stats.squared_error == 5);
}

} // namespace

int main() try {
  check_arithmetic();
  check_shapes();
  check_matrix();
  check_verify();
  check_workload();
  check_error_stats();
  std::puts("Host regressions passed");
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
