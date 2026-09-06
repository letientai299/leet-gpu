#include "matmul.hpp"

#include <cstdint>
#include <cstdio>
#include <exception>
#include <string_view>
#include <vector>

namespace {

constexpr unsigned height = 67;
constexpr unsigned width = 33;
constexpr unsigned inner = 50;

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

void bench(nvbench::state& state, MatmulKernel launch) {
  unsigned bench_height = 0;
  unsigned bench_width = 0;
  unsigned bench_k = 0;
  if (get_matmul_dimensions(state, bench_height, bench_width, bench_k)) {
    benchmark_matmul(state, bench_height, bench_width, bench_k, launch);
  }
}

void bench_cell(nvbench::state& state) {
  bench(state, launch_matmul_cell);
}

void bench_row(nvbench::state& state) {
  bench(state, launch_matmul_row);
}

void bench_col(nvbench::state& state) {
  bench(state, launch_matmul_col);
}

#define MATMUL_BENCHMARK(name, function)                                                           \
  NVBENCH_BENCH(function)                                                                          \
      .set_name(name)                                                                              \
      .set_min_samples(20)                                                                         \
      .set_cold_warmup_runs(5)                                                                     \
      .set_batch_target_time(1.0)                                                                  \
      .set_throttle_threshold(0.9F)                                                                \
      .set_throttle_recovery_delay(0.1F)                                                           \
      .add_int64_axis("Height", {height})                                                          \
      .add_int64_axis("Width", {width})                                                            \
      .add_int64_axis("K", {inner})

MATMUL_BENCHMARK("matmul.cell", bench_cell);
MATMUL_BENCHMARK("matmul.row", bench_row);
MATMUL_BENCHMARK("matmul.col", bench_col);

#undef MATMUL_BENCHMARK

void print_usage(const char* app) {
  std::printf("Usage: %s [--kernel cell|row|col] [--bench [options]]\n", app);
}

} // namespace

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
    return run_matmul_kernel(height, width, inner, launch);
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
