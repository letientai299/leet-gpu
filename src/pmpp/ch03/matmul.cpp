#include "matmul.hpp"

#include "matmul/benchmark.hpp"

namespace mm = lg::matmul;

int main(int argc, char** argv) {
  return mm::run_app(argc, argv, {{"cell", kMatmulCell}, {"row", kMatmulRow}, {"col", kMatmulCol}});
}
