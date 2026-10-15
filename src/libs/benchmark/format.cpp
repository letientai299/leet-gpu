#include "benchmark/format.hpp"

#include "benchmark/report.hpp"
#include "revision.hpp"

#include <algorithm>
#include <fmt/format.h>
#include <string>
#include <string_view>

namespace lg::benchmark {
namespace {

constexpr int kModelWhatWidth = 24;
constexpr int kModelBytesWidth = 10;

void print_row(std::string_view label, const std::string& value, int width = kLabelWidth) {
  if (!value.empty()) {
    fmt::print("  {} {}\n", label_cell(label, ansi::dim, width), value);
  }
}

// Prints the section only when one of its rows has a value.
void print_block(std::string_view title, std::string_view note, const Rows& rows) {
  if (!std::ranges::any_of(rows, has_value)) {
    return;
  }
  int width = kLabelWidth;
  for (const auto& row : rows) {
    if (has_value(row)) {
      width = std::max(width, static_cast<int>(row.label.size()));
    }
  }
  print_section(title, note);
  for (const auto& row : rows) {
    print_row(row.label, row.cell.value + row.cell.detail, width);
  }
}

void print_model(const Model& model) {
  if (model.rows.empty()) {
    return;
  }
  const int bytes_end = 2 + kLabelWidth + 1 + kModelWhatWidth + kModelBytesWidth;
  constexpr std::string_view title = " model";
  fmt::print(
    "{}{}\n", paint(title, ansi::bold),
    paint(
      fmt::format("{:>{}}  {:>6}", "bytes", bytes_end - static_cast<int>(title.size()), "FLOP/B"),
      ansi::dim
    )
  );
  if (!model.work.empty()) {
    fmt::print("  Algorithmic work: {}\n", model.work);
  }
  for (const auto& row : model.rows) {
    fmt::print(
      "  {} {:<{}}{:>{}}  {:>6.2f}\n", label_cell(row.label), row.what, kModelWhatWidth,
      format_bytes(row.bytes), kModelBytesWidth, row.intensity
    );
  }
}

void print_meters(const std::vector<Meter>& meters) {
  if (meters.empty()) {
    return;
  }
  print_section("utilization", kMeterNote);
  for (const auto& meter : meters) {
    fmt::print(
      "  {} {}  {}\n", label_cell(meter.label, meter.tint), meter_value(meter), meter.detail
    );
  }
}

void print_report(const Report& report) {
  fmt::print(
    "\n{}  {}\n", paint(report.benchmark, ansi::bold),
    paint(
      fmt::format(
        "{}{}[{}] · build {}", report.gpu, report.gpu.empty() ? "" : " ", report.device,
        build_revision()
      ),
      ansi::dim
    )
  );
  for (const auto& row : report.skipped) {
    print_row(row.label, row.cell.value);
  }
  print_block("setup", {}, report.setup);
  print_model(report.model);
  print_block("measured", kMeasuredNote, report.measured);
  print_meters(report.meters);
}

} // namespace

void print_result(std::string_view input) {
  for (const auto& report : parse_reports(input, false)) {
    print_report(report);
  }
}

} // namespace lg::benchmark
