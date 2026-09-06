#include "matmul.hpp"

#include <cstdint>
#include <cstdio>
#include <exception>
#include <string_view>
#include <vector>

namespace {

enum class Kernel : std::uint8_t { cell, row, col };

struct AppArgs {
  Kernel kernel = Kernel::cell;
  bool bench = false;
  bool help = false;
  std::vector<char*> nvbench;
};

bool parse_kernel(std::string_view name, Kernel& kernel) {
  if (name == "cell") {
    kernel = Kernel::cell;
  } else if (name == "row") {
    kernel = Kernel::row;
  } else if (name == "col") {
    kernel = Kernel::col;
  } else {
    return false;
  }
  return true;
}

bool parse_args(int argc, char** argv, AppArgs& args) {
  args.nvbench.push_back(argv[0]);
  for (int index = 1; index < argc; ++index) {
    const std::string_view arg = argv[index];
    if (arg == "--bench") {
      args.bench = true;
      continue;
    }
    if (arg == "--help" || arg == "-h") {
      args.help = true;
      continue;
    }
    if (arg == "--kernel") {
      ++index;
      if (index == argc || !parse_kernel(argv[index], args.kernel)) {
        return false;
      }
      continue;
    }

    constexpr std::string_view prefix = "--kernel=";
    if (arg.rfind(prefix, 0) == 0) {
      if (!parse_kernel(arg.substr(prefix.size()), args.kernel)) {
        return false;
      }
      continue;
    }
    args.nvbench.push_back(argv[index]);
  }
  return args.bench || args.nvbench.size() == 1;
}

MatmulKernel get_kernel(Kernel kernel) {
  switch (kernel) {
  case Kernel::cell:
    return launch_matmul_cell;
  case Kernel::row:
    return launch_matmul_row;
  case Kernel::col:
    return launch_matmul_col;
  }
  return launch_matmul_cell;
}

char* get_benchmark(Kernel kernel) {
  static char cell[] = "matmul.cell";
  static char row[] = "matmul.row";
  static char col[] = "matmul.col";
  switch (kernel) {
  case Kernel::cell:
    return cell;
  case Kernel::row:
    return row;
  case Kernel::col:
    return col;
  }
  return cell;
}

void print_usage(const char* app) {
  std::printf("Usage: %s [--kernel cell|row|col] [--bench [options]]\n", app);
}

} // namespace

void matmul_nvbench(nvbench::state& state, MatmulKernel launch) {
  unsigned height = 0;
  unsigned width = 0;
  unsigned k = 0;
  if (get_matmul_dimensions(state, height, width, k)) {
    benchmark_matmul(state, height, width, k, launch);
  }
}

#define MATMUL_NVBENCH_AXES()                                                                      \
  .add_int64_axis("Height", {kMatmulHeight})                                                       \
      .add_int64_axis("Width", {kMatmulWidth})                                                     \
      .add_int64_axis("K", {kMatmulK})
#include "matmul.nvbench.hpp"

int main(int argc, char** argv) try {
  AppArgs args;
  if (!parse_args(argc, argv, args)) {
    print_usage(argv[0]);
    return 2;
  }
  if (args.help) {
    print_usage(argv[0]);
    return 0;
  }

  const auto launch = get_kernel(args.kernel);
  const int result = run_host(1, argv, [launch] {
    // Ragged dimensions require every kernel to bound-check.
    return run_matmul_kernel(kMatmulHeight, kMatmulWidth, kMatmulK, launch);
  });
  if (result != 0 || !args.bench) {
    return result;
  }

  static char benchmark_option[] = "--benchmark";
  args.nvbench.push_back(benchmark_option);
  args.nvbench.push_back(get_benchmark(args.kernel));
  return run_nvbench_args(static_cast<int>(args.nvbench.size()), args.nvbench.data());
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
