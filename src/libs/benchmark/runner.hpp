#pragma once

#include <cstddef>
#include <memory>
#include <nvbench/benchmark.cuh>
#include <nvbench/benchmark_manager.cuh>
#include <nvbench/nvbench.cuh>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace lg::benchmark {

/// Applies the shared measurement policy.
nvbench::benchmark_base& configure(nvbench::benchmark_base& entry);

/// Registers a runtime benchmark.
template <typename Runner> nvbench::benchmark_base& add(std::string_view name, Runner runner) {
  auto entry = std::make_unique<nvbench::benchmark<Runner>>(std::move(runner));
  return configure(
      nvbench::benchmark_manager::get().add(std::move(entry)).set_name(std::string(name)));
}

/// Adds an integer result column.
void add_summary(nvbench::state& state,
                 std::string tag,
                 std::string name,
                 nvbench::int64_t value,
                 std::string hint = {});

/// Adds throughput and convergence columns.
void finish_summaries(nvbench::state& state,
                      std::string_view prefix,
                      std::size_t flops,
                      std::optional<std::size_t> model_bytes = std::nullopt);

/// Runs NVBench with forwarded arguments.
int run_args(int argc, char** argv);

} // namespace lg::benchmark
