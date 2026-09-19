#pragma once

#include "matmul/kernel.hpp"

#include <initializer_list>
#include <memory>
#include <nvbench/main.cuh>
#include <nvbench/nvbench.cuh>
#include <optional>
#include <string_view>

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

struct AppConfig {
  Kernel kernel;
  Kernel baseline;
  GemmShape bench_shape;
};

struct KernelChoice {
  std::string_view option;
  Kernel kernel;
};

void benchmark(nvbench::state& state, GemmShape shape, Kernel kernel);
std::optional<GemmShape> get_shape(nvbench::state& state);
int run_nvbench_args(int argc, char** argv);
int run_app(int argc, char** argv, const AppConfig& config);
int run_app(int argc, char** argv, std::initializer_list<KernelChoice> choices);
int run_bench_app(int argc, char** argv, std::initializer_list<Kernel> kernels);

} // namespace lg::matmul
