#pragma once

#include "matmul/kernel.hpp"

#include <optional>
#include <vector>

namespace lg::matmul {

inline constexpr unsigned kDefaultHeight = 67;
inline constexpr unsigned kDefaultWidth = 33;
inline constexpr unsigned kDefaultK = 50;

/// Returns the standard correctness shape.
inline GemmShape default_shape() {
  return GemmShape(kDefaultHeight, kDefaultWidth, kDefaultK);
}

/// Shape flags are optional so a caller can tell "unset" from "set to the default".
struct AppArgs {
  std::optional<unsigned> height;
  std::optional<unsigned> width;
  std::optional<unsigned> k;
  bool bench = false;
  bool help = false;
  std::vector<char*> remaining;

  /// Resolves omitted dimensions from `fallback`.
  [[nodiscard]] GemmShape shape(const GemmShape& fallback = default_shape()) const;
};

/// Parses shared matrix application arguments.
bool parse_args(int argc, char** argv, AppArgs& args);
/// Initializes host and CUDA state.
bool start();
/// Runs one kernel correctness check.
int run_check(int argc, char** argv, Kernel kernel, GemmShape shape);
/// Prints correctness application usage.
void print_shape_usage(const char* app);
/// Prints benchmark application usage.
void print_bench_usage(const char* app);

} // namespace lg::matmul
