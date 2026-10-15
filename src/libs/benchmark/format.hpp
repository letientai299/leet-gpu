#pragma once

#include <span>
#include <string>
#include <string_view>

namespace lg::benchmark {

/// Prints captured NVBench JSON as grouped rows. Throws on malformed JSON.
void print_result(std::string_view input);

/// One captured run, named for the comparison.
struct Run {
  std::string label;
  std::string json;
};

/// Prints runs as stacked rows; the first run is the baseline for ratios.
/// Throws `std::invalid_argument` for fewer than two runs, `std::runtime_error`
/// when the runs differ in state count or axis values, and the JSON library's
/// exception for malformed input.
void print_comparison(std::span<const Run> runs);

} // namespace lg::benchmark
