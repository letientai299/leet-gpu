#pragma once

#include "benchmark/resources.cuh"

#include <cstddef>
#include <memory>
#include <nvbench/benchmark.cuh>
#include <nvbench/benchmark_manager.cuh>
#include <nvbench/nvbench.cuh>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lg::benchmark {

/// Applies the shared measurement policy.
nvbench::benchmark_base& configure(nvbench::benchmark_base& entry);

/// Registers a runtime benchmark.
template <typename Runner> nvbench::benchmark_base& add(std::string_view name, Runner runner) {
  auto entry = std::make_unique<nvbench::benchmark<Runner>>(std::move(runner));
  return configure(
    nvbench::benchmark_manager::get().add(std::move(entry)).set_name(std::string(name))
  );
}

/// Adds an integer result column.
void add_summary(
  nvbench::state& state,
  std::string tag,
  std::string name,
  nvbench::int64_t value,
  std::string hint = {}
);

/// Adds concrete compiled-kernel and launch resources.
void add_resources(nvbench::state& state, const KernelResources& resources);

/// Bytes moved through global memory.
struct MemoryTraffic {
  std::size_t read_bytes = 0;
  std::size_t write_bytes = 0;
};

/// What one benchmark state computes and moves; every app reports the same fields.
struct Workload {
  std::vector<std::pair<std::string, nvbench::int64_t>> shape;
  std::size_t flops = 0;
  std::size_t buffer_bytes = 0;
  // Each input read once and each output written once.
  MemoryTraffic minimum;
  // Every operand access reaches DRAM: no cache or shared-memory reuse.
  std::optional<MemoryTraffic> no_reuse;
};

/// Converts float element counts to bytes.
std::optional<MemoryTraffic> float_traffic(std::size_t reads, std::size_t writes);

/// Adds workload columns before timing; skips the state when invalid.
bool describe(nvbench::state& state, const Workload& workload);

/// Adds throughput, utilization, and convergence columns.
void finish_summaries(nvbench::state& state, const Workload& workload);

/// Runs NVBench with forwarded arguments.
int run_args(int argc, char** argv);

} // namespace lg::benchmark
