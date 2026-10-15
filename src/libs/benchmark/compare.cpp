#include "benchmark/format.hpp"
#include "benchmark/report.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fmt/format.h>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace lg::benchmark {
namespace {

// Ratios inside this band, or inside the runs' combined noise, are not news.
constexpr double kFlatBand = 0.01;

std::string pad_right(std::string_view text, int width) {
  return std::string(text) +
         std::string(static_cast<std::size_t>(std::max(0, width - visible_width(text))), ' ');
}

std::string pad_left(std::string_view text, int width) {
  return std::string(static_cast<std::size_t>(std::max(0, width - visible_width(text))), ' ') +
         std::string(text);
}

// Green when better and red when worse, outside the noise band.
std::string ratio_text(const Cell& cell, const Cell& baseline) {
  if (!cell.number || !baseline.number || *baseline.number == 0.0) {
    return {};
  }
  const double factor = *cell.number / *baseline.number;
  const std::string text = fmt::format("{:.2f}×", factor);
  const double band =
    std::max(kFlatBand, std::hypot(cell.noise.value_or(0.0), baseline.noise.value_or(0.0)));
  if (cell.better == Better::unknown || std::fabs(factor - 1.0) < band) {
    return paint(text, ansi::dim);
  }
  const bool improved = (cell.better == Better::lower) == (factor < 1.0);
  return paint(text, improved ? ansi::green : ansi::red);
}

struct Line {
  std::size_t app;
  const Cell* cell;
};

// One row label, stacked across apps.
struct Group {
  std::string label;
  std::size_t occurrence;
  std::vector<const Cell*> cells; // per app; null when the app lacks the row
  std::vector<Line> lines;
  bool shared = false;
};

// Rows with the same label and occurrence stack, in order of first appearance.
std::vector<Group> group_rows(const std::vector<const Rows*>& per_app) {
  std::vector<Group> groups;
  for (std::size_t app = 0; app < per_app.size(); ++app) {
    std::map<std::string, std::size_t> seen;
    for (const auto& row : *per_app[app]) {
      if (!has_value(row)) {
        continue;
      }
      const std::size_t occurrence = seen[row.label]++;
      auto found = std::ranges::find_if(groups, [&](const auto& group) {
        return group.label == row.label && group.occurrence == occurrence;
      });
      if (found == groups.end()) {
        groups.push_back(
          {row.label, occurrence, std::vector<const Cell*>(per_app.size(), nullptr), {}}
        );
        found = groups.end() - 1;
      }
      found->cells[app] = &row.cell;
    }
  }
  return groups;
}

// Text rows that read the same for every app print once.
bool is_shared(const Group& group) {
  return group.cells.size() > 1 && std::ranges::all_of(group.cells, [&](const Cell* cell) {
           return cell != nullptr && !cell->number && cell->value == group.cells.front()->value &&
                  cell->detail == group.cells.front()->detail;
         });
}

// Apps without the row get `missing` when `fill` is set.
void place_lines(Group& group, const Cell& missing, bool fill) {
  group.shared = is_shared(group);
  const std::size_t shown = group.shared ? 1 : group.cells.size();
  for (std::size_t app = 0; app < shown; ++app) {
    const Cell* cell = group.cells[app] != nullptr ? group.cells[app] : (fill ? &missing : nullptr);
    if (cell != nullptr) {
      group.lines.push_back({app, cell});
    }
  }
}

std::string ratio_for(const Group& group, const Line& line) {
  if (line.app == 0 || group.cells[line.app] == nullptr || !line.cell->number) {
    return {};
  }
  if (group.cells.front() == nullptr) {
    return paint("–", ansi::dim);
  }
  return ratio_text(*line.cell, *group.cells.front());
}

struct Widths {
  int label = kLabelWidth;
  int app = 3;
  int value = 0;
  int ratio = 0;
};

Widths measure(const std::vector<Group>& groups, const std::vector<std::string>& apps) {
  Widths widths;
  for (const auto& group : groups) {
    widths.label = std::max(widths.label, visible_width(group.label));
    for (const auto& line : group.lines) {
      widths.app = std::max(widths.app, visible_width(apps[line.app]));
      widths.ratio = std::max(widths.ratio, visible_width(ratio_for(group, line)));
      // Text rows stay unpadded so a long status cannot widen the ratio column.
      if (line.cell->number) {
        widths.value = std::max(widths.value, visible_width(line.cell->value));
      }
    }
  }
  return widths;
}

void print_group(const Group& group, const std::vector<std::string>& apps, const Widths& widths) {
  for (const auto& line : group.lines) {
    const bool first = &line == &group.lines.front();
    const Cell& cell = *line.cell;
    std::string text = fmt::format(
      "  {} {} {}", label_cell(first ? group.label : "", ansi::dim, widths.label),
      pad_right(group.shared ? paint("all", ansi::dim) : apps[line.app], widths.app),
      pad_right(cell.value, cell.number ? widths.value : 0)
    );
    if (widths.ratio > 0 && cell.number) {
      text += " " + pad_left(ratio_for(group, line), widths.ratio);
    }
    text += cell.detail;
    fmt::print("{}\n", text.substr(0, text.find_last_not_of(' ') + 1));
  }
}

struct Block {
  std::string_view title;
  std::string_view note;
  bool fill_missing = true;
};

void print_compare_block(
  const Block& block, const std::vector<std::string>& apps, const std::vector<const Rows*>& per_app
) {
  auto groups = group_rows(per_app);
  if (groups.empty()) {
    return;
  }
  const Cell missing{paint("n/a", ansi::dim), {}, std::nullopt};
  for (auto& group : groups) {
    place_lines(group, missing, block.fill_missing);
  }
  if (!block.title.empty()) {
    print_section(block.title, block.note);
  }
  const Widths widths = measure(groups, apps);
  for (const auto& group : groups) {
    print_group(group, apps, widths);
  }
}

// Bytes carry no ratio: the minimum belongs to the problem, not the kernel.
Rows model_rows(const Model& model) {
  Rows rows;
  if (!model.work.empty()) {
    rows.push_back({"Work", {model.work}});
  }
  for (const auto& row : model.rows) {
    rows.push_back(
      {row.label, {format_bytes(row.bytes), fmt::format("  {:.2f} FLOP/B", row.intensity)}}
    );
  }
  return rows;
}

Rows meter_rows(const std::vector<Meter>& meters) {
  Rows rows;
  rows.reserve(meters.size());
  for (const auto& meter : meters) {
    rows.push_back({meter.label, {meter_value(meter), "  " + meter.detail, meter.fraction}});
  }
  return rows;
}

std::string joined_names(const std::vector<Report>& reports) {
  std::string names = reports.front().benchmark;
  for (const auto& report : reports) {
    if (report.benchmark != reports.front().benchmark) {
      names += " vs " + report.benchmark;
    }
  }
  return names;
}

template <typename Items, typename Project>
std::vector<const Rows*> collect(const Items& items, Project project) {
  std::vector<const Rows*> rows;
  rows.reserve(items.size());
  for (const auto& item : items) {
    rows.push_back(project(item));
  }
  return rows;
}

std::optional<double> clock_of(const Report& report) {
  const auto found = std::ranges::find(report.measured, "Clock", &Row::label);
  return found == report.measured.end() ? std::nullopt : found->cell.number;
}

// Apps run one after another, each under its own clock lock.
bool clocks_differ(const std::vector<Report>& reports) {
  const auto base = clock_of(reports.front());
  return base && std::ranges::any_of(reports, [&](const Report& report) {
           const auto other = clock_of(report);
           return other && std::fabs(*other / *base - 1.0) > kFlatBand;
         });
}

void print_state(const std::vector<std::string>& apps, const std::vector<Report>& reports) {
  const auto& first = reports.front();
  fmt::print(
    "\n{}  {}\n", paint(joined_names(reports), ansi::bold),
    paint(fmt::format("{}{}[{}]", first.gpu, first.gpu.empty() ? "" : " ", first.device), ansi::dim)
  );
  std::vector<Rows> models;
  std::vector<Rows> meters;
  for (const auto& report : reports) {
    models.push_back(model_rows(report.model));
    meters.push_back(meter_rows(report.meters));
  }
  const auto rows = [](const Rows& rows) {
    return &rows;
  };
  print_compare_block({{}, {}, false}, apps, collect(reports, [](const Report& report) {
                        return &report.skipped;
                      }));
  print_compare_block({"setup"}, apps, collect(reports, [](const Report& report) {
                        return &report.setup;
                      }));
  print_compare_block({"model", "bytes · FLOP/B"}, apps, collect(models, rows));
  print_compare_block({"measured", kMeasuredNote}, apps, collect(reports, [](const Report& report) {
                        return &report.measured;
                      }));
  if (clocks_differ(reports)) {
    fmt::print(
      "  {}\n", paint("note: SM clocks differ; time ratios include the clock change", ansi::yellow)
    );
  }
  print_compare_block({"utilization", kMeterNote}, apps, collect(meters, rows));
}

// Runs pair by position, so a different shape would compare unrelated states.
void check_pairing(
  const std::vector<std::string>& apps, const std::vector<std::vector<Report>>& runs
) {
  const auto& base = runs.front();
  for (std::size_t app = 1; app < runs.size(); ++app) {
    if (runs[app].size() != base.size()) {
      throw std::runtime_error(
        fmt::format(
          "{} has {} benchmark states; {} has {}", apps[app], runs[app].size(), apps[0], base.size()
        )
      );
    }
    for (std::size_t state = 0; state < base.size(); ++state) {
      if (runs[app][state].axes != base[state].axes) {
        throw std::runtime_error(
          fmt::format(
            "axis values differ at state {} ({}): {} vs {}", state + 1, base[state].benchmark,
            apps[0], apps[app]
          )
        );
      }
    }
  }
}

} // namespace

void print_comparison(std::span<const Run> runs) {
  if (runs.size() < 2) {
    throw std::invalid_argument("comparison needs at least two runs");
  }
  std::vector<std::string> apps;
  std::vector<std::vector<Report>> parsed;
  for (const auto& run : runs) {
    apps.push_back(run.label);
    parsed.push_back(parse_reports(run.json, true));
  }
  check_pairing(apps, parsed);
  for (std::size_t state = 0; state < parsed.front().size(); ++state) {
    std::vector<Report> reports;
    reports.reserve(parsed.size());
    for (const auto& run : parsed) {
      reports.push_back(run[state]);
    }
    print_state(apps, reports);
  }
}

} // namespace lg::benchmark
