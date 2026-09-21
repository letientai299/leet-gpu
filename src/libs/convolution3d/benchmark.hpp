#pragma once

#include "convolution3d/kernel.hpp"

namespace lg::convolution3d {

inline constexpr Shape kDefaultShape{33, 29, 23, 3};
inline constexpr Shape kDefaultBenchShape{256, 256, 128, 3};

int run_app(int argc, char** argv, Kernel kernel, Shape bench_shape = kDefaultBenchShape);

} // namespace lg::convolution3d
