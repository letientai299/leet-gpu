#include "benchmark/report.hpp"

#include "log.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <set>
#include <utility>

namespace lg::benchmark {
namespace {

using Json = nlohmann::json;

struct Unit {
  double scale;
  std::string_view name;
};

// Largest unit first.
using Units = std::array<Unit, 4>;

constexpr int kBarWidth = 20;

constexpr Units kSeconds{{{1.0, "s"}, {1.0e-3, "ms"}, {1.0e-6, "us"}, {1.0e-9, "ns"}}};
constexpr Units kBytes{
  {{1024.0 * 1024.0 * 1024.0, "GiB"}, {1024.0 * 1024.0, "MiB"}, {1024.0, "KiB"}, {1.0, "B"}}};
constexpr Units kByteRate{{{1.0e12, "TB/s"}, {1.0e9, "GB/s"}, {1.0e6, "MB/s"}, {1.0e3, "kB/s"}}};
constexpr Units kItemRate{{{1.0e12, "T/s"}, {1.0e9, "G/s"}, {1.0e6, "M/s"}, {1.0e3, "k/s"}}};
constexpr Units kFlops{{{1.0e12, "TFLOP"}, {1.0e9, "GFLOP"}, {1.0e6, "MFLOP"}, {1.0e3, "kFLOP"}}};
constexpr Units kFlopRate{
  {{1.0e12, "TFLOP/s"}, {1.0e9, "GFLOP/s"}, {1.0e6, "MFLOP/s"}, {1.0e3, "kFLOP/s"}}};
constexpr Units kHertz{{{1.0e9, "GHz"}, {1.0e6, "MHz"}, {1.0e3, "kHz"}, {1.0, "Hz"}}};

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

std::string raw_value(const Json& item) {
  const auto& value = item.at("value");
  return value.is_string() ? value.get<std::string>() : value.dump();
}

const Json* find_value(const Json& summary) {
  if (!summary.contains("data") || !summary.at("data").is_array()) {
    return nullptr;
  }
  const auto& data = summary.at("data");
  const auto found = std::ranges::find_if(data, [](const auto& item) {
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
      if (summary->value("tag", "").starts_with(prefix)) {
        used_.insert(summary);
        found.emplace_back(summary->value("name", ""), format_value(*summary));
      }
    }
    return found;
  }

  [[nodiscard]] std::vector<const Json*> rest() const {
    std::vector<const Json*> rest;
    for (const Json* summary : entries_) {
      if (!used_.contains(summary) && !summary->contains("hide")) {
        rest.push_back(summary);
      }
    }
    return rest;
  }

private:
  std::vector<const Json*> entries_;
  std::set<const Json*> used_;
};

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

void add_model_row(
  Summaries& summaries,
  Model& model,
  std::string_view label,
  std::string_view what,
  const std::string& base
) {
  const auto bytes = summaries.number(base + "/bytes");
  const auto intensity = summaries.number(base + "/intensity");
  if (!bytes) {
    return;
  }
  model.rows.push_back({std::string(label), std::string(what), *bytes, intensity.value_or(0.0)});
}

Model collect_model(Summaries& summaries) {
  Model model;
  if (summaries.find("lg/traffic/minimum/bytes") == nullptr) {
    return model;
  }
  model.work = summaries.text("lg/work/flops");
  add_model_row(summaries, model, "Minimum", "each value once", "lg/traffic/minimum");
  add_model_row(summaries, model, "No reuse", "every operand from DRAM", "lg/traffic/no_reuse");
  return model;
}

// One row per timer: mean, noise, then the spread in the mean's unit. Stacked
// rows repeat the unit, since neighbors may use another one.
Cell timing_cell(
  Summaries& summaries, const std::string& base, std::string_view count_tag, bool stacked
) {
  const auto mean = summaries.number(base + "/mean");
  if (!mean) {
    return {};
  }
  const auto [scale, unit] = pick_unit(*mean, kSeconds);
  const auto in_unit = [divisor = scale, unit = unit, stacked](double value) {
    return stacked ? fmt::format("{:.2f} {}", value / divisor, unit)
                   : fmt::format("{:.2f}", value / divisor);
  };

  Cell cell{fmt::format("{:.2f} {}", *mean / scale, unit), {}, mean, Better::lower};
  if (const auto noise = summaries.number(base + "/stdev/relative")) {
    cell.detail += fmt::format(" ±{:.2f}%", *noise * 100.0);
    cell.noise = noise;
  }
  for (const auto& [suffix, label] :
       {std::pair{"/min", "min"}, {"/median", "med"}, {"/max", "max"}}) {
    if (const auto value = summaries.number(base + suffix)) {
      cell.detail += fmt::format("  {} {}", label, in_unit(*value));
    }
  }
  if (const auto count = count_tag.empty() ? std::nullopt : summaries.number(count_tag)) {
    cell.detail += fmt::format("  ×{:.0f}", *count);
  }
  summaries.with_prefix(base + "/");
  return cell;
}

Cell clock_cell(Summaries& summaries) {
  Cell cell{summaries.text("nv/cold/sm_clock_rate/mean"), {}, std::nullopt};
  cell.number = summaries.number("nv/cold/sm_clock_rate/mean");
  // NVBench divides by cudaDevAttrClockRate, the maximum boost clock.
  const auto scaling = summaries.number("nv/cold/sm_clock_rate/scaling/percent");
  if (!cell.value.empty() && scaling) {
    cell.detail = fmt::format(" · {:.0f}% of max", *scaling * 100.0);
  }
  return cell;
}

std::string status_value(Summaries& summaries) {
  const std::string status = summaries.text("lg/cold/status");
  if (status.empty()) {
    return {};
  }
  const char* tint = ansi::yellow;
  if (status.starts_with("stable") || status.starts_with("single run")) {
    tint = ansi::green;
  } else if (status.starts_with("TIMED OUT") || status.starts_with("INSUFFICIENT") ||
             status.starts_with("NO SAMPLES") || status.starts_with("THROTTLED") ||
             status.starts_with("UNHEALTHY")) {
    tint = ansi::red;
  }
  return paint(status, tint);
}

Better better_for(std::string_view hint) {
  if (hint == "duration" || hint == "bytes") {
    return Better::lower;
  }
  if (hint == "byte_rate" || hint == "item_rate") {
    return Better::higher;
  }
  return Better::unknown;
}

Cell summary_cell(const Json& summary) {
  Cell cell{format_value(summary), {}, std::nullopt, better_for(summary.value("hint", ""))};
  const auto* item = find_value(summary);
  const std::string type = item == nullptr ? std::string{} : item->value("type", "");
  if (type == "int64" || type == "float64") {
    const double value = std::stod(raw_value(*item));
    if (std::isfinite(value)) {
      cell.number = value;
    }
  }
  return cell;
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

std::string ratio(double used, double peak, const Units& units) {
  const auto [scale, unit] = pick_unit(peak, units);
  return fmt::format("{:.2f} / {:.2f} {}", used / scale, peak / scale, unit);
}

std::string percent(double fraction) {
  return fmt::format("{:.2f}%", fraction * 100.0);
}

void compute_meters(Summaries& summaries, std::vector<Meter>& meters) {
  for (const auto& [kind, label] :
       std::array{std::pair{"fp32", "FP32"}, std::pair{"fp64", "FP64"}, std::pair{"fp16", "FP16"},
                  std::pair{"tensor", "Tensor"}, std::pair{"integer", "Integer"},
                  std::pair{"atomics", "Atomics"}, std::pair{"reductions", "Atom RED"},
                  std::pair{"shared_atomics", "Atom SMEM"}}) {
    const auto tag = std::string("lg/hardware/") + kind;
    const auto utilization = summaries.number(tag + "/utilization");
    if (!utilization) {
      continue;
    }
    const auto rate = summaries.number(tag + "/rate");
    const auto peak = summaries.number(tag + "/peak");
    std::string detail;
    if (rate && peak) {
      detail = ratio(*rate, *peak, kFlopRate);
    } else if (std::string_view(kind) == "integer" && rate) {
      detail = fmt::format("{:.3g} thread inst/s · ALU pipeline", *rate);
    } else if (rate) {
      detail = fmt::format("{:.3g} L2 sectors/s", *rate);
    } else if (std::string_view(kind) == "shared_atomics") {
      detail = "shared atomic warp instruction throughput";
    } else if (std::string_view(kind) == "fp16") {
      detail = "FP16 instruction throughput";
    } else {
      detail = "pipeline activity";
    }
    meters.push_back(
      {label, bar(*utilization), percent(*utilization), detail, *utilization, ansi::magenta}
    );
  }
}

std::vector<Meter> collect_meters(Summaries& summaries) {
  std::vector<Meter> meters;
  const auto dram_rate = summaries.number("lg/hardware/dram/rate");
  const auto dram_peak = summaries.number("lg/hardware/dram/peak");
  if (const auto dram = summaries.number("lg/hardware/dram/utilization")) {
    meters.push_back(
      {"DRAM", bar(*dram), percent(*dram),
       dram_rate && dram_peak ? ratio(*dram_rate, *dram_peak, kByteRate) : std::string{}, *dram,
       ansi::cyan}
    );
  }
  compute_meters(summaries, meters);
  const std::string warps = summaries.text("kernel/resources/warps_per_sm");
  const std::string max_warps = summaries.text("kernel/resources/max_warps_per_sm");
  const std::string limit = warps.empty() || max_warps.empty()
                              ? std::string{}
                              : fmt::format("{}/{} warps/SM", warps, max_warps);
  const auto theoretical = summaries.number("kernel/resources/occupancy");
  // Shading spans the gap up to the theoretical limit.
  if (const auto achieved = summaries.number("lg/occupancy/achieved")) {
    std::string detail = "achieved";
    if (theoretical) {
      append(detail, fmt::format("{} theoretical", percent(*theoretical)));
    }
    append(detail, limit);
    meters.push_back(
      {"Occupancy", range_bar(*achieved, theoretical.value_or(*achieved)), percent(*achieved),
       detail, *achieved, ansi::blue}
    );
  } else if (theoretical) {
    std::string detail = "theoretical";
    append(detail, limit);
    meters.push_back(
      {"Occupancy", bar(*theoretical), percent(*theoretical), detail, *theoretical, ansi::blue}
    );
  }
  return meters;
}

std::string device_name(const Json& root, int id) {
  if (!root.contains("devices") || !root.at("devices").is_array()) {
    return {};
  }
  const auto& devices = root.at("devices");
  const auto found = std::ranges::find_if(devices, [id](const auto& device) {
    return device.value("id", -1) == id;
  });
  return found == devices.end() ? std::string{} : found->value("name", "");
}

Report build_report(const Json& root, const Json& bench, const Json& state, bool stacked) {
  Report report;
  report.benchmark = bench.value("name", "Benchmark");
  report.axes = state.value("axis_values", Json::array()).dump();
  report.device = state.value("device", -1);
  report.gpu = device_name(root, report.device);
  if (state.value("is_skipped", false)) {
    report.skipped.push_back(
      {"Status", {paint("skipped: " + state.value("skip_reason", "unknown reason"), ansi::red)}}
    );
  }

  Summaries summaries(state);
  report.setup = {{"Problem", {problem_row(summaries, state)}},
                  {"Launch", {launch_row(summaries)}},
                  {"Compiled", {compiled_row(summaries)}}};
  report.model = collect_model(summaries);
  report.measured = {
    {"Cold", timing_cell(summaries, "nv/cold/time/gpu", "nv/cold/sample_size", stacked)},
    {"Batch", timing_cell(summaries, "nv/batch/time/gpu", "nv/batch/sample_size", stacked)},
    {"Host", timing_cell(summaries, "nv/cold/time/cpu", {}, stacked)},
    {"Clock", clock_cell(summaries)},
    {"Status", {status_value(summaries)}}};
  summaries.with_prefix("lg/batch/");
  summaries.with_prefix("nv/cold/walltime");
  summaries.with_prefix("nv/batch/walltime");

  report.meters = collect_meters(summaries);
  for (const Json* summary : summaries.rest()) {
    report.measured.push_back(
      {summary->value("name", summary->value("tag", "")), summary_cell(*summary)}
    );
  }
  return report;
}

} // namespace

std::string paint(std::string_view text, const char* tint) {
  static const bool color = use_color(stdout);
  if (!color) {
    return std::string(text);
  }
  return fmt::format("{}{}{}", tint, text, ansi::reset);
}

std::string format_bytes(double value) {
  return std::fabs(value) < 1024.0 ? fmt::format("{:.0f} B", value) : scaled(value, kBytes);
}

std::string label_cell(std::string_view label, const char* tint, int width) {
  return paint(fmt::format("{:<{}}", label, width), tint);
}

void print_section(std::string_view title, std::string_view note) {
  fmt::print(
    " {}{}\n", paint(title, ansi::bold),
    note.empty() ? std::string{} : "  " + paint(note, ansi::dim)
  );
}

std::string meter_value(const Meter& meter) {
  return fmt::format(
    "{} {}", paint(meter.bar, meter.tint), paint(fmt::format("{:>8}", meter.value), meter.tint)
  );
}

int visible_width(std::string_view text) {
  int width = 0;
  for (std::size_t index = 0; index < text.size(); ++index) {
    if (text[index] == '\033') {
      index = text.find('m', index);
      if (index == std::string_view::npos) {
        break;
      }
    } else if ((static_cast<unsigned char>(text[index]) & 0xC0U) != 0x80U) {
      ++width;
    }
  }
  return width;
}

std::vector<Report> parse_reports(std::string_view json, bool stacked) {
  const Json root = Json::parse(json);
  std::vector<Report> reports;
  for (const auto& bench : root.at("benchmarks")) {
    for (const auto& state : bench.at("states")) {
      reports.push_back(build_report(root, bench, state, stacked));
    }
  }
  return reports;
}

} // namespace lg::benchmark
