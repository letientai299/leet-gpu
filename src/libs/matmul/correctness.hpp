#pragma once

#include "matmul/kernel.hpp"

namespace lg::matmul {

inline constexpr float kAbsoluteTolerance = 1.0e-5F;
inline constexpr float kRelativeTolerance = 1.0e-4F;

/// Computes the CUTLASS reference result.
bool run_reference(Problem& problem);
/// Computes the candidate kernel result.
bool run_kernel(Problem& problem, Kernel kernel);
/// Compares candidate and reference results.
[[nodiscard]] bool verify(const Problem& problem);
/// Runs the complete correctness check.
int check(Problem& problem, Kernel kernel);

} // namespace lg::matmul
