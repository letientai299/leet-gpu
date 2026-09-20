#pragma once

#include "convolution/kernel.hpp"

namespace lg::convolution {

inline constexpr Shape kDefaultBenchShape{4096, 4096, 3};

/// Checks and benchmarks one kernel.
int run_app(int argc, char** argv, Kernel kernel, Shape bench_shape = kDefaultBenchShape);

} // namespace lg::convolution
