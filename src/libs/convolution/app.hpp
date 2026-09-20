#pragma once

#include "convolution/kernel.hpp"

namespace lg::convolution {

inline constexpr Shape kDefaultShape{257, 193, 3};

int run_check(int argc, char** argv, Kernel kernel, Shape shape = kDefaultShape);

} // namespace lg::convolution
