#pragma once

#include "benchmark/resources.cuh"
#include "benchmark/workload.hpp"
#include "checks.hpp"

#include <functional>
#include <memory>
#include <nvbench/benchmark.cuh>
#include <nvbench/benchmark_manager.cuh>
#include <nvbench/nvbench.cuh>
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

/// Adds the resources `query` reports; skips the state when it fails.
bool add_resources(
  nvbench::state& state, const std::function<cudaError_t(KernelResources&)>& query
);

/// Adds workload columns before timing; skips the state when invalid.
bool describe(nvbench::state& state, const Workload& workload);

/// Adds throughput, utilization, and convergence columns.
void finish_summaries(nvbench::state& state, const Workload& workload);

/// Reports resources and workload, then times `launch(stream)` on NVBench's stream.
template <typename Shape, typename Launch>
void measure(
  nvbench::state& state,
  const std::optional<Workload>& workload,
  cudaError_t (*resources)(const Shape&, KernelResources&),
  const Shape& shape,
  Launch&& launch
) {
  if (!CUDA_CHECK(cudaDeviceSynchronize())) {
    state.skip("CUDA setup failed");
    return;
  }
  if (!workload) {
    state.skip("workload overflow");
    return;
  }
  if (resources != nullptr && !add_resources(state, [resources, &shape](KernelResources& found) {
        return resources(shape, found);
      })) {
    return;
  }
  if (!describe(state, *workload)) {
    return;
  }
#ifdef __clang_analyzer__
  launch(cudaStream_t{});
#else
  state.exec(nvbench::exec_tag::gpu, [&](nvbench::launch& timed) {
    launch(timed.get_stream());
  });
  finish_summaries(state, *workload);
#endif
}

/// Runs NVBench with forwarded arguments.
int run_args(int argc, char** argv);

} // namespace lg::benchmark
