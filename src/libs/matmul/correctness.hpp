#pragma once

#include "matmul/kernel.hpp"

namespace lg::matmul {

inline constexpr float kAbsoluteTolerance = 1.0e-5F;
inline constexpr float kRelativeTolerance = 1.0e-4F;

bool run_reference(Problem& problem);
bool run_kernel(Problem& problem, Kernel kernel);
[[nodiscard]] bool verify(const Problem& problem);
int check(Problem& problem, Kernel kernel);

} // namespace lg::matmul
