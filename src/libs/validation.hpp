#pragma once

#include "log.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>

namespace lg {

constexpr double kDefaultAtol = 1.0e-5;
constexpr double kDefaultRtol = 1.0e-4;

constexpr std::optional<std::size_t> checked_mul(std::size_t left, std::size_t right) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
    return std::nullopt;
  }
  return left * right;
}

constexpr std::optional<std::size_t> checked_add(std::size_t left, std::size_t right) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    return std::nullopt;
  }
  return left + right;
}

constexpr std::size_t valid_axis_pairs(int length, int radius) {
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

/// Compares `checks` positions picked by `index_of(sample)`.
template <typename IndexOf>
bool verify_at(
  std::span<const float> actual,
  std::span<const float> expected,
  std::size_t checks,
  IndexOf index_of,
  const char* name,
  double atol,
  double rtol
) {
  ErrorStats stats;
  for (std::size_t sample = 0; sample < checks; ++sample) {
    const auto index = index_of(sample);
    stats.add(actual[index], expected[index], atol, rtol);
  }
  HOST_LOG(
    "{}: mismatches={}/{} nonfinite={}", name, stats.mismatches, stats.count, stats.nonfinite
  );
  HOST_LOG(
    "Finite errors: max_abs={:g} max_rel={:g} rms={:g}", stats.max_absolute, stats.max_relative,
    std::sqrt(
      stats.squared_error /
      static_cast<double>(std::max(std::size_t{1}, stats.count - stats.nonfinite))
    )
  );
  if (stats.mismatches != 0) {
    const auto index = index_of(stats.first);
    HOST_LOG(
      "First mismatch[{}]: actual={:g} expected={:g} atol={:g} rtol={:g}", index,
      static_cast<double>(actual[index]), static_cast<double>(expected[index]), atol, rtol
    );
  }
  return stats.mismatches == 0;
}

inline bool
same_size(std::span<const float> actual, std::span<const float> expected, const char* name) {
  if (actual.size() != expected.size() || actual.size() == 0) {
    HOST_LOG("{} invalid result size: {}/{}", name, actual.size(), expected.size());
    return false;
  }
  return true;
}

inline bool verify_values(
  std::span<const float> actual,
  std::span<const float> expected,
  const char* name,
  double atol = kDefaultAtol,
  double rtol = kDefaultRtol
) {
  return same_size(actual, expected, name) && verify_at(
                                                actual, expected, actual.size(),
                                                [](std::size_t index) {
                                                  return index;
                                                },
                                                name, atol, rtol
                                              );
}

/// Spreads `checks` positions evenly over `count` values, ends included.
inline std::size_t sample_index(std::size_t sample, std::size_t count, std::size_t checks) {
  return checks <= 1 || checks == count
           ? sample
           : static_cast<std::size_t>(
               static_cast<long double>(sample) * static_cast<long double>(count - 1) /
               static_cast<long double>(checks - 1)
             );
}

/// Compares every value when samples is 0. Otherwise requires every value to be
/// finite, then compares `samples` evenly spaced positions.
inline bool verify_sampled(
  std::span<const float> actual,
  std::span<const float> expected,
  std::size_t samples,
  const char* name
) {
  if (samples == 0) {
    return verify_values(actual, expected, name);
  }
  if (!same_size(actual, expected, name)) {
    return false;
  }
  const auto count = actual.size();
  for (std::size_t index = 0; index < count; ++index) {
    if (!std::isfinite(actual[index])) {
      HOST_LOG("{} nonfinite output[{}]", name, index);
      return false;
    }
  }
  const auto checks = std::min(samples, count);
  HOST_LOG("Reference coverage: {}/{} values", checks, count);
  return verify_at(
    actual, expected, checks,
    [count, checks](std::size_t sample) {
      return sample_index(sample, count, checks);
    },
    name, kDefaultAtol, kDefaultRtol
  );
}

} // namespace lg
