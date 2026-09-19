#include "matmul.hpp"

#include "matmul/benchmark.hpp"

int main(int argc, char** argv) {
  return lg::matmul::run_bench_app(argc, argv, {kMatmulCell, kMatmulRow, kMatmulCol});
}
