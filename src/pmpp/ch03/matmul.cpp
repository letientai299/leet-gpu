#include "matmul.hpp"

#include "matmul/app.hpp"
#include "matmul/benchmark.hpp"
#include "matmul/correctness.hpp"

#include <cstdint>
#include <cstdio>
#include <exception>
#include <string_view>
#include <vector>

namespace {

namespace mm = lg::matmul;

enum class KernelId : std::uint8_t { cell, row, col };

bool parse_kernel(std::string_view name, KernelId& kernel) {
  if (name == "cell") {
    kernel = KernelId::cell;
  } else if (name == "row") {
    kernel = KernelId::row;
  } else if (name == "col") {
    kernel = KernelId::col;
  } else {
    return false;
  }
  return true;
}

bool parse_kernel_args(int argc, char** argv, KernelId& kernel, std::vector<char*>& remaining) {
  remaining.push_back(argv[0]);
  for (int index = 1; index < argc; ++index) {
    const std::string_view arg = argv[index];
    if (arg == "--kernel") {
      ++index;
      if (index == argc || !parse_kernel(argv[index], kernel)) {
        return false;
      }
      continue;
    }
    constexpr std::string_view prefix = "--kernel=";
    if (arg.rfind(prefix, 0) == 0) {
      if (!parse_kernel(arg.substr(prefix.size()), kernel)) {
        return false;
      }
      continue;
    }
    remaining.push_back(argv[index]);
  }
  return true;
}

mm::Kernel get_kernel(KernelId kernel) {
  switch (kernel) {
  case KernelId::cell:
    return kMatmulCell;
  case KernelId::row:
    return kMatmulRow;
  case KernelId::col:
    return kMatmulCol;
  }
  return kMatmulCell;
}

char* get_benchmark(KernelId kernel) {
  static char cell[] = "matmul.cell";
  static char row[] = "matmul.row";
  static char col[] = "matmul.col";
  switch (kernel) {
  case KernelId::cell:
    return cell;
  case KernelId::row:
    return row;
  case KernelId::col:
    return col;
  }
  return cell;
}

void print_usage(const char* app) {
  std::printf("Usage: %s [--kernel cell|row|col] [--height N] [--width N] [--k N] "
              "[--bench [options]]\n",
              app);
  std::printf("Defaults: --height %u --width %u --k %u\n", mm::kDefaultHeight, mm::kDefaultWidth,
              mm::kDefaultK);
}

} // namespace

void matmul_nvbench(nvbench::state& state, lg::matmul::Kernel kernel) {
  const auto shape = lg::matmul::get_shape(state);
  if (shape.has_value()) {
    lg::matmul::benchmark(state, *shape, kernel);
  }
}

#define MATMUL_NVBENCH_AXES()                                                                      \
  .add_int64_axis("Height", {lg::matmul::kDefaultHeight})                                          \
      .add_int64_axis("Width", {lg::matmul::kDefaultWidth})                                        \
      .add_int64_axis("K", {lg::matmul::kDefaultK})
#include "matmul.nvbench.hpp"

int main(int argc, char** argv) try {
  KernelId kernel_id = KernelId::cell;
  std::vector<char*> common_args;
  if (!parse_kernel_args(argc, argv, kernel_id, common_args)) {
    print_usage(argv[0]);
    return 2;
  }

  mm::AppArgs args;
  if (!mm::parse_args(static_cast<int>(common_args.size()), common_args.data(), args) ||
      (!args.bench && args.remaining.size() != 1)) {
    print_usage(argv[0]);
    return 2;
  }
  if (args.help) {
    print_usage(argv[0]);
    return 0;
  }

  const auto kernel = get_kernel(kernel_id);
  if (!mm::start()) {
    return 1;
  }
  mm::Problem problem(args.shape());
  mm::fill_random(problem);
  const int result = mm::check(problem, kernel);
  if (result != 0 || !args.bench) {
    return result;
  }

  static char benchmark_option[] = "--benchmark";
  args.remaining.push_back(benchmark_option);
  args.remaining.push_back(get_benchmark(kernel_id));
  return mm::run_nvbench_args(static_cast<int>(args.remaining.size()), args.remaining.data());
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
