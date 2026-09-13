#include "matmul/app.hpp"

#include "checks.hpp"

#include <charconv>
#include <cstdio>
#include <limits>
#include <string_view>

namespace lg::matmul {
namespace {

bool parse_dimension(std::string_view text, unsigned& value) {
  unsigned parsed = 0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || parsed == 0 ||
      parsed > static_cast<unsigned>(std::numeric_limits<int>::max())) {
    return false;
  }
  value = parsed;
  return true;
}

bool parse_value(int& index, int argc, char** argv, unsigned& value) {
  ++index;
  return index < argc && parse_dimension(argv[index], value);
}

} // namespace

GemmShape AppArgs::shape() const {
  return GemmShape(height, width, k);
}

bool parse_args(int argc, char** argv, AppArgs& args) {
  args.remaining.push_back(argv[0]);
  for (int index = 1; index < argc; ++index) {
    const std::string_view arg = argv[index];
    if (arg == "--bench") {
      args.bench = true;
      continue;
    }
    if (arg == "--help" || arg == "-h") {
      args.help = true;
      continue;
    }
    if (arg == "--height") {
      if (!parse_value(index, argc, argv, args.height)) {
        return false;
      }
      continue;
    }
    if (arg == "--width") {
      if (!parse_value(index, argc, argv, args.width)) {
        return false;
      }
      continue;
    }
    if (arg == "--k") {
      if (!parse_value(index, argc, argv, args.k)) {
        return false;
      }
      continue;
    }

    constexpr std::string_view height_prefix = "--height=";
    constexpr std::string_view width_prefix = "--width=";
    constexpr std::string_view k_prefix = "--k=";
    if (arg.rfind(height_prefix, 0) == 0) {
      if (!parse_dimension(arg.substr(height_prefix.size()), args.height)) {
        return false;
      }
    } else if (arg.rfind(width_prefix, 0) == 0) {
      if (!parse_dimension(arg.substr(width_prefix.size()), args.width)) {
        return false;
      }
    } else if (arg.rfind(k_prefix, 0) == 0) {
      if (!parse_dimension(arg.substr(k_prefix.size()), args.k)) {
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
