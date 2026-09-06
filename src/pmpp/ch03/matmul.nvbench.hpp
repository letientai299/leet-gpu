#pragma once

#include "matmul.hpp"

#ifndef MATMUL_NVBENCH_AXES
#define MATMUL_NVBENCH_AXES()
#endif

void matmul_nvbench(nvbench::state& state, MatmulKernel launch);

inline void bench_matmul_cell(nvbench::state& state) {
  matmul_nvbench(state, launch_matmul_cell);
}

inline void bench_matmul_row(nvbench::state& state) {
  matmul_nvbench(state, launch_matmul_row);
}

inline void bench_matmul_col(nvbench::state& state) {
  matmul_nvbench(state, launch_matmul_col);
}

#define MATMUL_BENCHMARK(name, function)                                                           \
  NVBENCH_BENCH(function)                                                                          \
      .set_name(name)                                                                              \
      .set_min_samples(20)                                                                         \
      .set_cold_warmup_runs(5)                                                                     \
      .set_batch_target_time(1.0)                                                                  \
      .set_throttle_threshold(0.9F)                                                                \
      .set_throttle_recovery_delay(0.1F) MATMUL_NVBENCH_AXES()

MATMUL_BENCHMARK("matmul.cell", bench_matmul_cell);
MATMUL_BENCHMARK("matmul.row", bench_matmul_row);
MATMUL_BENCHMARK("matmul.col", bench_matmul_col);

#undef MATMUL_BENCHMARK
#undef MATMUL_NVBENCH_AXES
