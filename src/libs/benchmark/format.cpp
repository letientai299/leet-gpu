#include "benchmark/format.hpp"

#include "log.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lg::benchmark {
namespace {

using Json = nlohmann::json;

struct Unit {
  double scale;
  std::string_view name;
};

// Largest unit first.
using Units = std::array<Unit, 4>;

constexpr int kLabelWidth = 10;
constexpr int kBarWidth = 20;
constexpr int kModelWhatWidth = 24;
constexpr int kModelBytesWidth = 10;

constexpr Units kSeconds{{{1.0, "s"}, {1.0e-3, "ms"}, {1.0e-6, "us"}, {1.0e-9, "ns"}}};
constexpr Units kBytes{
  {{1024.0 * 1024.0 * 1024.0, "GiB"}, {1024.0 * 1024.0, "MiB"}, {1024.0, "KiB"}, {1.0, "B"}}};
constexpr Units kByteRate{{{1.0e12, "TB/s"}, {1.0e9, "GB/s"}, {1.0e6, "MB/s"}, {1.0e3, "kB/s"}}};
constexpr Units kItemRate{{{1.0e12, "T/s"}, {1.0e9, "G/s"}, {1.0e6, "M/s"}, {1.0e3, "k/s"}}};
constexpr Units kFlops{{{1.0e12, "TFLOP"}, {1.0e9, "GFLOP"}, {1.0e6, "MFLOP"}, {1.0e3, "kFLOP"}}};
constexpr Units kFlopRate{
  {{1.0e12, "TFLOP/s"}, {1.0e9, "GFLOP/s"}, {1.0e6, "MFLOP/s"}, {1.0e3, "kFLOP/s"}}};
constexpr Units kHertz{{{1.0e9, "GHz"}, {1.0e6, "MHz"}, {1.0e3, "kHz"}, {1.0, "Hz"}}};

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

std::string paint(std::string_view text, const char* tint) {
  if (!log_color()) {
    return std::string(text);
  }
  return fmt::format("{}{}{}", tint, text, ansi::reset);
}

bool starts_with(std::string_view text, std::string_view prefix) {
  return text.substr(0, prefix.size()) == prefix;
}

Unit pick_unit(double value, const Units& units) {
  for (const auto& unit : units) {
    if (std::fabs(value) >= unit.scale) {
      return unit;
    }
  }
  return units.back();
}

std::string scaled(double value, const Units& units, int precision = 1) {
  const auto [scale, unit] = pick_unit(value, units);
  return fmt::format("{:.{}f} {}", value / scale, precision, unit);
}

std::string format_bytes(double value) {
  return std::fabs(value) < 1024.0 ? fmt::format("{:.0f} B", value) : scaled(value, kBytes);
}

std::string raw_value(const Json& item) {
  const auto& value = item.at("value");
  return value.is_string() ? value.get<std::string>() : value.dump();
}

const Json* find_value(const Json& summary) {
  if (!summary.contains("data") || !summary.at("data").is_array()) {
    return nullptr;
  }
  const auto& data = summary.at("data");
  const auto found = std::find_if(data.begin(), data.end(), [](const auto& item) {
    return item.value("name", "") == "value";
  });
  return found == data.end() ? nullptr : &*found;
}

std::string format_value(const Json& summary) {
  const auto* item = find_value(summary);
  if (item == nullptr) {
    return {};
  }
  std::string raw = raw_value(*item);
  const std::string hint = summary.value("hint", "");
  if (hint.empty()) {
    if (item->value("type", "") == "float64") {
      return fmt::format("{:.5g}", std::stod(raw));
    }
    return raw;
  }

  const double value = std::stod(raw);
  if (!std::isfinite(value)) {
    return "n/a";
  }
  if (hint == "duration") {
    return scaled(value, kSeconds, 3);
  }
  if (hint == "frequency") {
    return scaled(value, kHertz, 3);
  }
  if (hint == "bytes") {
    return format_bytes(value);
  }
  if (hint == "byte_rate") {
    return scaled(value, kByteRate);
  }
  if (hint == "item_rate") {
    return scaled(value, kItemRate);
  }
  if (hint == "flops") {
    return scaled(value, kFlops);
  }
  if (hint == "sample_size") {
    return fmt::format("{}x", std::stoll(raw));
  }
  if (hint == "percentage") {
    return fmt::format("{:.2f}%", value * 100.0);
  }
  return raw;
}

// Looks up summaries by exact tag and tracks which ones a row consumed.
class Summaries {
public:
  explicit Summaries(const Json& state) {
    if (!state.contains("summaries") || !state.at("summaries").is_array()) {
      return;
    }
    for (const auto& summary : state.at("summaries")) {
      entries_.push_back(&summary);
    }
  }

  const Json* find(std::string_view tag) {
    for (const Json* summary : entries_) {
      if (summary->value("tag", "") == tag) {
        used_.insert(summary);
        return summary;
      }
    }
    return nullptr;
  }

  std::optional<double> number(std::string_view tag) {
    const Json* summary = find(tag);
    const Json* item = summary == nullptr ? nullptr : find_value(*summary);
    if (item == nullptr) {
      return std::nullopt;
    }
    const double value = std::stod(raw_value(*item));
    return std::isfinite(value) ? std::optional{value} : std::nullopt;
  }

  std::string text(std::string_view tag) {
    const Json* summary = find(tag);
    return summary == nullptr ? std::string{} : format_value(*summary);
  }

  // Returns {name, value} pairs whose tag starts with the prefix.
  std::vector<std::pair<std::string, std::string>> with_prefix(std::string_view prefix) {
    std::vector<std::pair<std::string, std::string>> found;
    for (const Json* summary : entries_) {
      if (starts_with(summary->value("tag", ""), prefix)) {
        used_.insert(summary);
        found.emplace_back(summary->value("name", ""), format_value(*summary));
      }
    }
    return found;
  }

  [[nodiscard]] std::vector<const Json*> rest() const {
    std::vector<const Json*> rest;
    for (const Json* summary : entries_) {
      if (used_.count(summary) == 0 && !summary->contains("hide")) {
        rest.push_back(summary);
      }
    }
    return rest;
  }

private:
  std::vector<const Json*> entries_;
  std::set<const Json*> used_;
};

std::string label_cell(std::string_view label, const char* tint = ansi::dim) {
  return paint(fmt::format("{:<{}}", label, kLabelWidth), tint);
}

void print_section(std::string_view title, std::string_view note = {}) {
  fmt::print(
    " {}{}\n", paint(title, ansi::bold),
    note.empty() ? std::string{} : "  " + paint(note, ansi::dim)
  );
}

void print_row(std::string_view label, const std::string& value) {
  if (!value.empty()) {
    fmt::print("  {} {}\n", label_cell(label), value);
  }
}

using Rows = std::vector<std::pair<std::string_view, std::string>>;

// Prints the section only when one of its rows has a value.
void print_block(std::string_view title, std::string_view note, const Rows& rows) {
  const bool any = std::any_of(rows.begin(), rows.end(), [](const auto& row) {
    return !row.second.empty();
  });
  if (!any) {
    return;
  }
  print_section(title, note);
  for (const auto& [label, value] : rows) {
    print_row(label, value);
  }
}

void append(std::string& out, const std::string& part, std::string_view separator = " · ") {
  if (part.empty()) {
    return;
  }
  if (!out.empty()) {
    out += separator;
  }
  out += part;
}

std::string suffixed(const std::string& value, std::string_view unit) {
  return value.empty() ? std::string{} : fmt::format("{} {}", value, unit);
}

std::string prefixed(std::string_view label, const std::string& value) {
  return value.empty() ? std::string{} : fmt::format("{} {}", label, value);
}

// Dimensions (upper case) join with ×, parameters (lower case) with ·.
std::string problem_row(Summaries& summaries, const Json& state) {
  std::string out;
  for (const auto& [name, value] : summaries.with_prefix("lg/shape/")) {
    const bool dimension = !name.empty() && std::isupper(static_cast<unsigned char>(name[0])) != 0;
    append(out, fmt::format("{} {}", name, value), dimension ? " × " : " · ");
  }
  if (out.empty() && state.contains("axis_values") && state.at("axis_values").is_array()) {
    for (const auto& axis : state.at("axis_values")) {
      append(out, axis.value("name", "Axis") + " " + raw_value(axis));
    }
  }
  append(out, summaries.text("lg/work/flops"));
  append(out, suffixed(summaries.text("lg/work/buffers"), "buffers"));
  return out;
}

std::string launch_row(Summaries& summaries) {
  std::string out = prefixed("block", summaries.text("kernel/resources/block"));
  const std::string threads = summaries.text("kernel/resources/threads_per_block");
  if (!threads.empty()) {
    out += fmt::format(" ({} thr)", threads);
  }
  append(out, suffixed(summaries.text("kernel/resources/blocks_per_sm"), "blk/SM"));
  return out;
}

std::string compiled_row(Summaries& summaries) {
  std::string out = suffixed(summaries.text("kernel/resources/registers_per_thread"), "reg/thr");
  append(out, prefixed("smem", summaries.text("kernel/resources/shared_static")));
  append(out, prefixed("dyn", summaries.text("kernel/resources/shared_dynamic")));
  append(out, prefixed("cmem", summaries.text("kernel/resources/constant")));
  append(out, prefixed("spill", summaries.text("kernel/resources/local_per_thread")));
  return out;
}

void print_model_row(
  Summaries& summaries, std::string_view label, std::string_view what, const std::string& base
) {
  const auto bytes = summaries.number(base + "/bytes");
  const auto intensity = summaries.number(base + "/intensity");
  if (!bytes) {
    return;
  }
  fmt::print(
    "  {} {:<{}}{:>{}}  {:>6.2f}\n", label_cell(label), what, kModelWhatWidth, format_bytes(*bytes),
    kModelBytesWidth, intensity.value_or(0.0)
  );
}

void print_model(Summaries& summaries) {
  if (summaries.find("lg/traffic/minimum/bytes") == nullptr) {
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
  print_model_row(summaries, "Minimum", "each value once", "lg/traffic/minimum");
  print_model_row(summaries, "No reuse", "every operand from DRAM", "lg/traffic/no_reuse");
}

// One row per timer: mean, noise, then the spread in the mean's unit.
std::string timing_row(Summaries& summaries, const std::string& base, std::string_view count_tag) {
  const auto mean = summaries.number(base + "/mean");
  if (!mean) {
    return {};
  }
  const auto [scale, unit] = pick_unit(*mean, kSeconds);
  const auto in_unit = [divisor = scale](double value) {
    return fmt::format("{:.2f}", value / divisor);
  };

  std::string out = fmt::format("{} {}", in_unit(*mean), unit);
  if (const auto noise = summaries.number(base + "/stdev/relative")) {
    out += fmt::format(" ±{:.2f}%", *noise * 100.0);
  }
  for (const auto& [suffix, label] :
       {std::pair{"/min", "min"}, {"/median", "med"}, {"/max", "max"}}) {
    if (const auto value = summaries.number(base + suffix)) {
      out += fmt::format("  {} {}", label, in_unit(*value));
    }
  }
  if (const auto count = count_tag.empty() ? std::nullopt : summaries.number(count_tag)) {
    out += fmt::format("  ×{:.0f}", *count);
  }
  summaries.with_prefix(base + "/");
  return out;
}

std::string clock_row(Summaries& summaries) {
  std::string out = summaries.text("nv/cold/sm_clock_rate/mean");
  // NVBench divides by cudaDevAttrClockRate, the maximum boost clock.
  const auto scaling = summaries.number("nv/cold/sm_clock_rate/scaling/percent");
  if (!out.empty() && scaling) {
    out += fmt::format(" · {:.0f}% of max", *scaling * 100.0);
  }
  return out;
}

std::string status_value(Summaries& summaries) {
  const std::string status = summaries.text("lg/cold/status");
  if (status.empty()) {
    return {};
  }
  const char* tint = ansi::yellow;
  if (starts_with(status, "stable") || starts_with(status, "single run")) {
    tint = ansi::green;
  } else if (starts_with(status, "TIMED OUT") || starts_with(status, "INSUFFICIENT") ||
             starts_with(status, "NO SAMPLES") || starts_with(status, "THROTTLED") ||
             starts_with(status, "UNHEALTHY")) {
    tint = ansi::red;
  }
  return paint(status, tint);
}

std::string cells(int count, std::string_view glyph) {
  std::string out;
  for (int cell = 0; cell < count; ++cell) {
    out += glyph;
  }
  return out;
}

std::string bar(double fraction) {
  constexpr std::array<std::string_view, 8> partial{"", "▏", "▎", "▍", "▌", "▋", "▊", "▉"};
  const int eighths = static_cast<int>(std::lround(std::clamp(fraction, 0.0, 1.0) * kBarWidth * 8));
  const int used = eighths / 8 + (eighths % 8 == 0 ? 0 : 1);
  return cells(eighths / 8, "█") + std::string(partial[static_cast<std::size_t>(eighths % 8)]) +
         cells(kBarWidth - used, " ");
}

// Solid up to the lower bound, shaded up to the upper bound.
std::string range_bar(double low, double high) {
  const auto to_cells = [](double fraction) {
    return static_cast<int>(std::lround(std::clamp(fraction, 0.0, 1.0) * kBarWidth));
  };
  const int solid = to_cells(low);
  const int shaded = std::max(0, to_cells(high) - solid);
  return cells(solid, "█") + cells(shaded, "▒") + cells(kBarWidth - solid - shaded, " ");
}

struct Meter {
  std::string_view label;
  std::string bar;
  std::string value;
  std::string detail;
  const char* tint;
};

std::string ratio(double used, double peak, const Units& units) {
  const auto [scale, unit] = pick_unit(peak, units);
  return fmt::format("{:.2f} / {:.2f} {}", used / scale, peak / scale, unit);
}

std::string percent(double fraction) {
  return fmt::format("{:.2f}%", fraction * 100.0);
}

std::vector<Meter> collect_meters(Summaries& summaries) {
  std::vector<Meter> meters;
  const auto cache_low = summaries.number("lg/cache/served_min");
  const auto cache_high = summaries.number("lg/cache/served_max");
  if (cache_low && cache_high) {
    meters.push_back(
      {"Reuse", range_bar(*cache_low, *cache_high),
       fmt::format("{:.0f}–{:.0f}%", *cache_low * 100.0, *cache_high * 100.0),
       "of no-reuse bytes served on chip", ansi::green}
    );
  }
  const auto dram_rate = summaries.number("nv/cold/bw/global/bytes_per_second");
  const auto dram_peak = summaries.number("lg/dram/peak");
  if (const auto dram = summaries.number("nv/cold/bw/global/utilization")) {
    meters.push_back(
      {"DRAM", bar(*dram), percent(*dram),
       dram_rate && dram_peak ? ratio(*dram_rate, *dram_peak, kByteRate) : std::string{},
       ansi::cyan}
    );
  }
  const auto gflops = summaries.number("lg/cold/gflops");
  const auto fp32_peak = summaries.number("lg/fp32/peak");
  if (const auto fp32 = summaries.number("lg/cold/fp32/utilization")) {
    meters.push_back(
      {"FP32", bar(*fp32), percent(*fp32),
       gflops && fp32_peak ? ratio(*gflops * 1.0e9, *fp32_peak, kFlopRate) : std::string{},
       ansi::magenta}
    );
  }
  const std::string warps = summaries.text("kernel/resources/warps_per_sm");
  const std::string max_warps = summaries.text("kernel/resources/max_warps_per_sm");
  if (const auto occupancy = summaries.number("kernel/resources/occupancy")) {
    meters.push_back(
      {"Occupancy", bar(*occupancy), percent(*occupancy),
       warps.empty() || max_warps.empty() ? std::string{}
                                          : fmt::format("{}/{} warps/SM", warps, max_warps),
       ansi::blue}
    );
  }
  return meters;
}

void print_meters(const std::vector<Meter>& meters) {
  if (meters.empty()) {
    return;
  }
  print_section("utilization", "cold runs; occupancy is theoretical");
  for (const auto& meter : meters) {
    fmt::print(
      "  {} {} {}  {}\n", label_cell(meter.label, meter.tint), paint(meter.bar, meter.tint),
      paint(fmt::format("{:>8}", meter.value), meter.tint), meter.detail
    );
  }
}

std::string device_name(const Json& root, int id) {
  if (!root.contains("devices") || !root.at("devices").is_array()) {
    return {};
  }
  const auto& devices = root.at("devices");
  const auto found = std::find_if(devices.begin(), devices.end(), [id](const auto& device) {
    return device.value("id", -1) == id;
  });
  return found == devices.end() ? std::string{} : found->value("name", "");
}

void print_state(const Json& root, const Json& bench, const Json& state) {
  const int device = state.value("device", -1);
  const std::string gpu = device_name(root, device);
  fmt::print(
    "\n{}  {}\n", paint(bench.value("name", "Benchmark"), ansi::bold),
    paint(fmt::format("{}{}[{}]", gpu, gpu.empty() ? "" : " ", device), ansi::dim)
  );
  if (state.value("is_skipped", false)) {
    print_row(
      "Status", paint("skipped: " + state.value("skip_reason", "unknown reason"), ansi::red)
    );
  }

  Summaries summaries(state);
  print_block(
    "setup", {},
    {{"Problem", problem_row(summaries, state)},
     {"Launch", launch_row(summaries)},
     {"Compiled", compiled_row(summaries)}}
  );
  print_model(summaries);
  print_block(
    "measured", "cold: L2 flushed per run · batch: back-to-back",
    {{"Cold", timing_row(summaries, "nv/cold/time/gpu", "nv/cold/sample_size")},
     {"Batch", timing_row(summaries, "nv/batch/time/gpu", "nv/batch/sample_size")},
     {"Host", timing_row(summaries, "nv/cold/time/cpu", {})},
     {"Clock", clock_row(summaries)},
     {"Status", status_value(summaries)}}
  );
  summaries.with_prefix("lg/batch/");
  summaries.with_prefix("nv/cold/walltime");
  summaries.with_prefix("nv/batch/walltime");

  const auto meters = collect_meters(summaries);
  for (const Json* summary : summaries.rest()) {
    print_row(summary->value("name", summary->value("tag", "")), format_value(*summary));
  }
  print_meters(meters);
}

} // namespace

void print_result(std::string_view input) {
  const Json root = Json::parse(input);
  for (const auto& bench : root.at("benchmarks")) {
    for (const auto& state : bench.at("states")) {
      print_state(root, bench, state);
    }
  }
}

} // namespace lg::benchmark
