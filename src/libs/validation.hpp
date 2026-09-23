#pragma once

#include "log.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>

namespace lg {

inline std::optional<std::size_t> checked_mul(std::size_t left, std::size_t right) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
    return std::nullopt;
  }
  return left * right;
}

inline std::optional<std::size_t> checked_add(std::size_t left, std::size_t right) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    return std::nullopt;
  }
  return left + right;
}

inline std::size_t valid_axis_pairs(int length, int radius) {
  if (length <= 0 || radius < 0) {
    throw std::invalid_argument("invalid convolution dimensions");
  }
  const auto n = static_cast<std::size_t>(length);
  const auto r = static_cast<std::size_t>(std::min(radius, length - 1));
  return n * (2 * r + 1) - r * (r + 1);
}

struct ErrorStats {
  std::size_t count = 0;
  std::size_t mismatches = 0;
  std::size_t nonfinite = 0;
  std::size_t first = 0;
  double max_absolute = 0;
  double max_relative = 0;
  double squared_error = 0;

  void add(float actual, float expected, double atol, double rtol) {
    const bool finite = std::isfinite(actual) && std::isfinite(expected);
    const double difference = std::fabs(static_cast<double>(actual) - expected);
    if (!finite || difference > atol + rtol * std::fabs(expected)) {
      if (mismatches == 0) {
        first = count;
      }
      ++mismatches;
    }
    ++count;
    if (!finite) {
      ++nonfinite;
      return;
    }
    max_absolute = std::max(max_absolute, difference);
    const double relative = expected == 0.0F
                              ? (difference == 0 ? 0 : std::numeric_limits<double>::infinity())
                              : difference / std::fabs(expected);
    max_relative = std::max(max_relative, relative);
    squared_error += difference * difference;
  }
};

template <typename Actual, typename Expected>
bool verify_values(
  const Actual& actual,
  const Expected& expected,
  const char* name,
  double atol = 1.0e-5,
  double rtol = 1.0e-4
) {
  if (actual.size() != expected.size() || actual.size() == 0) {
    HOST_LOG("%s invalid result size: %zu/%zu", name, actual.size(), expected.size());
    return false;
  }
  ErrorStats stats;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    stats.add(actual.data()[index], expected.data()[index], atol, rtol);
  }
  HOST_LOG(
    "%s: mismatches=%zu/%zu nonfinite=%zu", name, stats.mismatches, stats.count, stats.nonfinite
  );
  HOST_LOG(
    "Finite errors: max_abs=%g max_rel=%g rms=%g", stats.max_absolute, stats.max_relative,
    std::sqrt(
      stats.squared_error /
      static_cast<double>(std::max(std::size_t{1}, stats.count - stats.nonfinite))
    )
  );
  if (stats.mismatches != 0) {
    HOST_LOG(
      "First mismatch[%zu]: actual=%g expected=%g atol=%g rtol=%g", stats.first,
      actual.data()[stats.first], expected.data()[stats.first], atol, rtol
    );
  }
  return stats.mismatches == 0;
}

} // namespace lg
