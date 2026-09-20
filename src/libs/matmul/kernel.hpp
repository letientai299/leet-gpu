#pragma once

#include "benchmark/resources.cuh"
#include "matmul/matrix.hpp"

#include <cstddef>
#include <cuda_runtime_api.h>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace lg::matmul {

using KernelCallback = void (*)(const float*, const float*, float*, const GemmShape&, cudaStream_t);
using MatrixTransform = Matrix (*)(const Matrix&);

struct InputTransforms {
  MatrixTransform a = nullptr;
  MatrixTransform b = nullptr;

  [[nodiscard]] bool operator==(const InputTransforms& other) const {
    return a == other.a && b == other.b;
  }
};

/// Owns any kernel-specific input layouts.
class KernelInputs {
public:
  KernelInputs(const Problem& problem, InputTransforms transforms)
      : problem_(problem), transformed_a_(apply(transforms.a, problem.a)),
        transformed_b_(apply(transforms.b, problem.b)) {
    if (a().size() != problem.a.size() || b().size() != problem.b.size()) {
      throw std::invalid_argument("kernel input transform changed element count");
    }
  }

  [[nodiscard]] const Matrix& a() const {
    return transformed_a_ ? *transformed_a_ : problem_.a;
  }

  [[nodiscard]] const Matrix& b() const {
    return transformed_b_ ? *transformed_b_ : problem_.b;
  }

private:
  static std::optional<Matrix> apply(MatrixTransform transform, const Matrix& input) {
    return transform == nullptr ? std::nullopt : std::optional<Matrix>{transform(input)};
  }

  const Problem& problem_;
  std::optional<Matrix> transformed_a_;
  std::optional<Matrix> transformed_b_;
};

/// Multiplies counts without overflow.
inline std::optional<std::size_t> checked_mul(std::size_t left, std::size_t right) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
    return std::nullopt;
  }
  return left * right;
}

/// Adds counts without overflow.
inline std::optional<std::size_t> checked_add(std::size_t left, std::size_t right) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    return std::nullopt;
  }
  return left + right;
}

/// Bytes a kernel moves through global memory, for NVBench bandwidth columns.
struct Traffic {
  std::size_t global_reads = 0;
  std::size_t global_writes = 0;
};

using TrafficCallback = bool (*)(const GemmShape&, Traffic&);
using ResourcesCallback = cudaError_t (*)(const GemmShape&,
                                          lg::benchmark::KernelResources& resources);

/// Estimates naive per-cell memory traffic.
inline bool cell_traffic(const GemmShape& shape, Traffic& traffic) {
  const auto reads = checked_mul(shape.c_size(), std::size_t{shape.k()} * 2);
  if (!reads) {
    return false;
  }
  traffic.global_reads = *reads;
  traffic.global_writes = shape.c_size();
  return true;
}

struct Kernel {
  std::string_view name;
  KernelCallback launch;
  TrafficCallback traffic = nullptr;
  InputTransforms inputs;
  ResourcesCallback resources = nullptr;
};

} // namespace lg::matmul
