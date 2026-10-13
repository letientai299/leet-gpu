#include "matmul.hpp"

#include "matmul/benchmark.hpp"

namespace mm = lg::matmul;

constexpr unsigned kBenchSize = 4096;

int main(int argc, char** argv) {
  return mm::run_app(
    argc, argv, {{"cell", kMatmulCell}, {"row", kMatmulRow}, {"col", kMatmulCol}},
    mm::GemmShape(kBenchSize, kBenchSize, kBenchSize)
  );
}
