#pragma once

#include "matmul/kernel.hpp"

#include <memory>
#include <nvbench/main.cuh>
#include <nvbench/nvbench.cuh>
#include <optional>

namespace lg::matmul {

class Benchmark {
public:
  explicit Benchmark(Problem problem);
  ~Benchmark();

  Benchmark(Benchmark&&) noexcept;
  Benchmark& operator=(Benchmark&&) noexcept;
  Benchmark(const Benchmark&) = delete;
  Benchmark& operator=(const Benchmark&) = delete;

  void run(nvbench::state& state, Kernel kernel);
  void run_with_shape(nvbench::state& state, Kernel kernel);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

void benchmark(nvbench::state& state, GemmShape shape, Kernel kernel);
std::optional<GemmShape> get_shape(nvbench::state& state);
int run_nvbench_args(int argc, char** argv);

} // namespace lg::matmul
