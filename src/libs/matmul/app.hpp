#pragma once

#include "matmul/matrix.hpp"

#include <vector>

namespace lg::matmul {

inline constexpr unsigned kDefaultHeight = 67;
inline constexpr unsigned kDefaultWidth = 33;
inline constexpr unsigned kDefaultK = 50;

struct AppArgs {
  unsigned height = kDefaultHeight;
  unsigned width = kDefaultWidth;
  unsigned k = kDefaultK;
  bool bench = false;
  bool help = false;
  std::vector<char*> remaining;

  [[nodiscard]] GemmShape shape() const;
};

bool parse_args(int argc, char** argv, AppArgs& args);
bool start();
void print_shape_usage(const char* app);
void print_bench_usage(const char* app);

} // namespace lg::matmul
