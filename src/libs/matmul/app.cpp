#include "matmul/app.hpp"

#include "checks.hpp"
#include "matmul/correctness.hpp"

#include <charconv>
#include <cstdio>
#include <string_view>

namespace lg::matmul {
namespace {

using DimensionField = std::optional<unsigned> AppArgs::*;

struct DimensionOption {
  std::string_view flag;
  DimensionField field;
};

constexpr DimensionOption kDimensions[] = {
    {"--height", &AppArgs::height},
    {"--width", &AppArgs::width},
    {"--k", &AppArgs::k},
};

/// std::string_view::starts_with is C++20; this file targets C++17.
bool starts_with(std::string_view text, std::string_view prefix) {
  return text.size() >= prefix.size() && text.substr(0, prefix.size()) == prefix;
}

bool parse_dimension(std::string_view text, std::optional<unsigned>& value) {
  unsigned parsed = 0;
  const auto* first = text.data();
  const auto* last = first + text.size();
  const auto result = std::from_chars(first, last, parsed);
  if (result.ec != std::errc{} || result.ptr != last || parsed == 0 || parsed > kMaxDimension) {
    return false;
  }
  value = parsed;
  return true;
}

/// Consumes `--flag value` or `--flag=value`. Returns nullopt when `arg` is not a dimension.
std::optional<bool>
parse_dimension_arg(std::string_view arg, int& index, int argc, char** argv, AppArgs& args) {
  for (const auto& option : kDimensions) {
    if (arg == option.flag) {
      ++index;
      return index < argc && parse_dimension(argv[index], args.*option.field);
    }
    if (starts_with(arg, option.flag) && arg[option.flag.size()] == '=') {
      return parse_dimension(arg.substr(option.flag.size() + 1), args.*option.field);
    }
  }
  return std::nullopt;
}

} // namespace

GemmShape AppArgs::shape(const GemmShape& fallback) const {
  return GemmShape(height.value_or(fallback.m()), width.value_or(fallback.n()),
                   k.value_or(fallback.k()));
}

bool parse_args(int argc, char** argv, AppArgs& args) {
  args.remaining.push_back(argv[0]);
  for (int index = 1; index < argc; ++index) {
    const std::string_view arg = argv[index];
    if (arg == "--bench") {
      args.bench = true;
    } else if (arg == "--help" || arg == "-h") {
      args.help = true;
    } else if (const auto parsed = parse_dimension_arg(arg, index, argc, argv, args)) {
      if (!*parsed) {
        return false;
      }
    } else {
      args.remaining.push_back(argv[index]);
    }
  }
  return true;
}

bool start() {
  return init_host();
}

int run_check(int argc, char** argv, Kernel kernel, GemmShape shape) {
  return run_host(argc, argv, [&] {
    Problem problem(shape);
    fill_random(problem);
    return check(problem, kernel);
  });
}

void print_shape_usage(const char* app) {
  std::printf("Usage: %s [--height N] [--width N] [--k N]\n", app);
  std::printf("Defaults: --height %u --width %u --k %u\n", kDefaultHeight, kDefaultWidth,
              kDefaultK);
}

void print_bench_usage(const char* app) {
  std::printf("Usage: %s [--height N] [--width N] [--k N] [NVBench options]\n", app);
  std::printf("Defaults: --height %u --width %u --k %u\n", kDefaultHeight, kDefaultWidth,
              kDefaultK);
}

} // namespace lg::matmul
