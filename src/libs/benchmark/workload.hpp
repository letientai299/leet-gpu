#pragma once

#include "validation.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lg::benchmark {

/// Global-memory element accesses a kernel performs.
struct Traffic {
  std::size_t global_reads = 0;
  std::size_t global_writes = 0;
};

/// Bytes moved through global memory.
struct MemoryTraffic {
  std::size_t read_bytes = 0;
  std::size_t write_bytes = 0;
};

/// What one benchmark state computes and moves; every app reports the same fields.
struct Workload {
  std::vector<std::pair<std::string, std::int64_t>> shape;
  std::size_t flops = 0;
  std::size_t buffer_bytes = 0;
  // Each input read once and each output written once.
  MemoryTraffic minimum;
  // Every operand access reaches DRAM: no cache or shared-memory reuse.
  std::optional<MemoryTraffic> no_reuse;
};

/// Converts float element counts to bytes.
inline std::optional<MemoryTraffic> float_traffic(std::size_t reads, std::size_t writes) {
  const auto read_bytes = checked_mul(reads, sizeof(float));
  const auto write_bytes = checked_mul(writes, sizeof(float));
  if (!read_bytes || !write_bytes) {
    return std::nullopt;
  }
  return MemoryTraffic{*read_bytes, *write_bytes};
}

/// Float workload whose buffers are exactly its minimum traffic.
inline std::optional<Workload> float_workload(
  std::vector<std::pair<std::string, std::int64_t>> shape,
  std::optional<std::size_t> flops,
  std::optional<std::size_t> reads,
  std::size_t writes
) {
  const auto minimum = reads ? float_traffic(*reads, writes) : std::nullopt;
  const auto buffers =
    minimum ? checked_add(minimum->read_bytes, minimum->write_bytes) : std::nullopt;
  if (!flops || !buffers) {
    return std::nullopt;
  }
  return Workload{std::move(shape), *flops, *buffers, *minimum, std::nullopt};
}

/// Sets `workload.no_reuse` from `traffic`; nullopt when the model fails.
inline std::optional<Workload>
with_traffic(std::optional<Workload> workload, const std::function<bool(Traffic&)>& traffic) {
  Traffic counts;
  if (!workload || !traffic(counts)) {
    return std::nullopt;
  }
  workload->no_reuse = float_traffic(counts.global_reads, counts.global_writes);
  return workload->no_reuse ? workload : std::nullopt;
}

/// Adds the kernel's no-reuse model when it has one; nullopt on overflow.
template <typename ShapeArg, typename Shape>
std::optional<Workload> with_no_reuse(
  std::optional<Workload> workload, bool (*traffic)(ShapeArg, Traffic&), const Shape& shape
) {
  if (traffic == nullptr) {
    return workload;
  }
  return with_traffic(std::move(workload), [traffic, &shape](Traffic& counts) {
    return traffic(shape, counts);
  });
}

} // namespace lg::benchmark
