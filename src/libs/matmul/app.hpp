#pragma once

#include "matmul/kernel.hpp"

#include <vector>

namespace lg::matmul {

inline constexpr unsigned kDefaultHeight = 67;
inline constexpr unsigned kDefaultWidth = 33;
inline constexpr unsigned kDefaultK = 50;

struct AppArgs {
  unsigned height = kDefaultHeight;
  unsigned width = kDefaultWidth;
  unsigned k = kDefaultK;
  bool has_height = false;
  bool has_width = false;
  bool has_k = false;
  bool bench = false;
  bool help = false;
  std::vector<char*> remaining;

  [[nodiscard]] GemmShape shape() const;
};

bool parse_args(int argc, char** argv, AppArgs& args);
bool start();
int run_check(int argc, char** argv, Kernel kernel, GemmShape shape);
void print_shape_usage(const char* app);
void print_bench_usage(const char* app);

} // namespace lg::matmul
