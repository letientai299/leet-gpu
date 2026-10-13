#pragma once

#include "stencil/kernel.hpp"

namespace lg::stencil {

inline constexpr Shape kDefaultShape{33, 29, 23};
inline constexpr Shape kDefaultBenchShape{256, 256, 256};

/// Checks and benchmarks one kernel.
int run_app(int argc, char** argv, Kernel kernel, Shape bench_shape = kDefaultBenchShape);

} // namespace lg::stencil
