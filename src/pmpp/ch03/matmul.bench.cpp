
#include "log.hpp"
#include "matmul/app.hpp"
#include "matmul/benchmark.hpp"

#include <cstdio>
#include <exception>
#include <memory>
#include <utility>

namespace {

namespace mm = lg::matmul;

std::unique_ptr<mm::Benchmark> harness;

void print_usage(const char* app) {
  mm::print_bench_usage(app);
}

} // namespace

void matmul_nvbench(nvbench::state& state, lg::matmul::Kernel kernel) {
  harness->run_with_shape(state, kernel);
}

#include "matmul.nvbench.hpp"

int main(int argc, char** argv) try {
  mm::AppArgs args;
  if (!mm::parse_args(argc, argv, args)) {
    print_usage(argv[0]);
    return 2;
  }
  if (args.help) {
    print_usage(argv[0]);
    static char help[] = "--help";
    args.remaining.push_back(help);
  }
  if (!mm::start()) {
    return 1;
  }

  mm::Problem problem(args.shape());
  mm::fill_random(problem);
  harness = std::make_unique<mm::Benchmark>(std::move(problem));
  HOST_LOG("Benchmark shape: height %u, width %u, k %u", args.height, args.width, args.k);
  const int result =
      mm::run_nvbench_args(static_cast<int>(args.remaining.size()), args.remaining.data());
  harness.reset();
  return result;
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
