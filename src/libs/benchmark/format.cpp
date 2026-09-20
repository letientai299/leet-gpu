#include "benchmark/format.hpp"

#include <algorithm>
#include <cmath>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lg::benchmark {
namespace {

using Json = nlohmann::json;
using Field = std::pair<std::string, std::string>;

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

std::string scaled(double value, std::initializer_list<std::pair<double, std::string_view>> units) {
  for (const auto& [scale, unit] : units) {
    if (std::fabs(value) >= scale) {
      return fmt::format("{:.3f} {}", value / scale, unit);
    }
  }
  return fmt::format("{:.3f}", value);
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
  if (hint == "duration") {
    return scaled(value, {{1.0, "s"}, {1.0e-3, "ms"}, {1.0e-6, "us"}, {1.0e-9, "ns"}});
  }
  if (hint == "frequency") {
    return scaled(value, {{1.0e9, "GHz"}, {1.0e6, "MHz"}, {1.0e3, "kHz"}});
  }
  if (hint == "bytes") {
    if (value == 0.0) {
      return "0 B";
    }
    return scaled(
      value,
      {{1024.0 * 1024.0 * 1024.0, "GiB"}, {1024.0 * 1024.0, "MiB"}, {1024.0, "KiB"}, {1.0, "B"}}
    );
  }
  if (hint == "byte_rate") {
    return scaled(value, {{1.0e12, "TB/s"}, {1.0e9, "GB/s"}, {1.0e6, "MB/s"}, {1.0e3, "kB/s"}});
  }
  if (hint == "item_rate") {
    return scaled(value, {{1.0e12, "T/s"}, {1.0e9, "G/s"}, {1.0e6, "M/s"}, {1.0e3, "k/s"}});
  }
  if (hint == "flops") {
    return scaled(
      value,
      {{1.0e12, "TFLOP"}, {1.0e9, "GFLOP"}, {1.0e6, "MFLOP"}, {1.0e3, "kFLOP"}, {1.0, "FLOP"}}
    );
  }
  if (hint == "sample_size") {
    return fmt::format("{}x", std::stoll(raw));
  }
  if (hint == "percentage") {
    return fmt::format("{:.2f}%", value * 100.0);
  }
  return raw;
}

std::string field_name(const Json& summary) {
  const std::string tag = summary.value("tag", "");
  std::string name = summary.value("name", tag);
  if (tag == "nv/cold/sample_size") {
    return "Cold Samples";
  }
  if (tag == "nv/cold/time/cpu/mean") {
    return "Cold CPU Time";
  }
  if (tag == "nv/cold/time/cpu/stdev/relative") {
    return "Cold CPU Noise";
  }
  if (tag == "nv/cold/time/gpu/mean") {
    return "Cold GPU Time";
  }
  if (tag == "nv/cold/time/gpu/stdev/relative") {
    return "Cold GPU Noise";
  }
  if (tag == "nv/cold/sm_clock_rate/mean") {
    return "Cold Clock Rate";
  }
  if (tag == "nv/cold/sm_clock_rate/scaling/percent") {
    return "Cold Clock Scaling";
  }
  if (tag == "nv/batch/sample_size") {
    return "Batch Samples";
  }
  return name;
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

void add_axes(std::vector<Field>& fields, const Json& state) {
  if (!state.contains("axis_values") || !state.at("axis_values").is_array()) {
    return;
  }
  for (const auto& axis : state.at("axis_values")) {
    fields.emplace_back(axis.value("name", "Axis"), raw_value(axis));
  }
}

void add_summaries(std::vector<Field>& fields, const Json& state) {
  if (!state.contains("summaries") || !state.at("summaries").is_array()) {
    return;
  }
  for (const auto& summary : state.at("summaries")) {
    if (summary.contains("hide")) {
      continue;
    }
    fields.emplace_back(field_name(summary), format_value(summary));
  }
}

void print_fields(const std::vector<Field>& fields) {
  const auto longest =
    std::max_element(fields.begin(), fields.end(), [](const auto& left, const auto& right) {
      return left.first.size() < right.first.size();
    });
  const auto width = longest == fields.end() ? std::size_t{0} : longest->first.size();
  for (const auto& [name, value] : fields) {
    fmt::print("  {:{}}  {}\n", name, width, value);
  }
}

} // namespace

void print_result(std::string_view input) {
  const Json root = Json::parse(input);
  fmt::print("\nBenchmark Results\n");
  for (const auto& bench : root.at("benchmarks")) {
    for (const auto& state : bench.at("states")) {
      const int device = state.value("device", -1);
      const std::string gpu = device_name(root, device);
      fmt::print(
        "\n{} [Device={}{}{}]\n", bench.value("name", "Benchmark"), device, gpu.empty() ? "" : ", ",
        gpu
      );

      if (state.value("is_skipped", false)) {
        fmt::print("  Status  Skipped: {}\n", state.value("skip_reason", "unknown reason"));
        continue;
      }

      std::vector<Field> fields;
      add_axes(fields, state);
      add_summaries(fields, state);
      print_fields(fields);
    }
  }
}

} // namespace lg::benchmark
