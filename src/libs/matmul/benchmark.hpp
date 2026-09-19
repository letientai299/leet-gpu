#pragma once

#include "matmul/kernel.hpp"

#include <initializer_list>
#include <memory>
#include <nvbench/nvbench.cuh>
#include <optional>
#include <string_view>

namespace lg::matmul {

/// Keeps one device copy of a problem alive across every NVBench state.
class Benchmark {
public:
  /// Owns one benchmark problem.
  explicit Benchmark(Problem problem);
  /// Releases cached device problems.
  ~Benchmark();

  /// Transfers benchmark ownership.
  Benchmark(Benchmark&&) noexcept;
  /// Replaces benchmark ownership.
  Benchmark& operator=(Benchmark&&) noexcept;
  Benchmark(const Benchmark&) = delete;
  Benchmark& operator=(const Benchmark&) = delete;

  /// Measures a kernel.
  void run(nvbench::state& state, Kernel kernel);
  /// Reports shape before measuring.
  void run_with_shape(nvbench::state& state, Kernel kernel);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// One kernel measured against a baseline at a fixed shape.
struct AppConfig {
  Kernel kernel;
  Kernel baseline;
  GemmShape bench_shape;
};

/// One kernel picked by `--kernel <option>`.
struct KernelChoice {
  std::string_view option;
  Kernel kernel;
};

/// Measures a generated problem.
void benchmark(nvbench::state& state, GemmShape shape, Kernel kernel);
/// Reads validated shape axes.
std::optional<GemmShape> get_shape(nvbench::state& state);
/// Runs NVBench with forwarded arguments.
int run_nvbench_args(int argc, char** argv);
/// Checks and benchmarks one configured kernel.
int run_app(int argc, char** argv, const AppConfig& config);
/// Checks and benchmarks one selected kernel.
int run_app(int argc, char** argv, std::initializer_list<KernelChoice> choices);
/// Benchmarks each supplied kernel.
int run_bench_app(int argc, char** argv, std::initializer_list<Kernel> kernels);

} // namespace lg::matmul
