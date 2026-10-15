#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace lg::benchmark {

namespace ansi {

constexpr auto reset = "\033[0m";
constexpr auto bold = "\033[1m";
constexpr auto dim = "\033[2m";
constexpr auto red = "\033[31m";
constexpr auto green = "\033[32m";
constexpr auto yellow = "\033[33m";
constexpr auto blue = "\033[34m";
constexpr auto magenta = "\033[35m";
constexpr auto cyan = "\033[36m";

} // namespace ansi

constexpr int kLabelWidth = 10;
constexpr std::string_view kMeasuredNote = "cold: L2 flushed per run · batch: back-to-back";
constexpr std::string_view kMeterNote = "hardware counters, separate cold replay";

/// Which direction of change counts as an improvement.
enum class Better : std::uint8_t { unknown, lower, higher };

/// One displayed value. `number` is what comparison ratios divide and `noise`
/// is its relative standard deviation.
struct Cell {
  std::string value;
  std::string detail;
  std::optional<double> number;
  Better better = Better::unknown;
  std::optional<double> noise;
};

struct Row {
  std::string label;
  Cell cell;
};

using Rows = std::vector<Row>;

struct ModelRow {
  std::string label;
  std::string what;
  double bytes;
  double intensity;
};

struct Model {
  std::string work;
  std::vector<ModelRow> rows;
};

struct Meter {
  std::string label;
  std::string bar;
  std::string value;
  std::string detail;
  double fraction;
  const char* tint;
};

/// Everything one benchmark state prints, so single and side-by-side output
/// share it.
struct Report {
  std::string benchmark;
  std::string axes;
  std::string gpu;
  int device = -1;
  Rows skipped;
  Rows setup;
  Model model;
  Rows measured;
  std::vector<Meter> meters;
};

inline bool has_value(const Row& row) {
  return !row.cell.value.empty();
}

/// Wraps text in an ANSI color when stdout is a terminal.
std::string paint(std::string_view text, const char* tint);

std::string format_bytes(double value);

std::string
label_cell(std::string_view label, const char* tint = ansi::dim, int width = kLabelWidth);

void print_section(std::string_view title, std::string_view note = {});

/// Bar, then percentage.
std::string meter_value(const Meter& meter);

/// Display columns: UTF-8 code points outside ANSI color sequences.
int visible_width(std::string_view text);

/// One report per benchmark state, in file order. `stacked` repeats the unit on
/// every timing statistic, because stacked rows no longer share one unit.
/// Throws on malformed JSON.
std::vector<Report> parse_reports(std::string_view json, bool stacked);

} // namespace lg::benchmark
