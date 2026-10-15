#include "benchmark/format.hpp"

#include <cstddef>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::string read_file(const std::string& path) {
  const std::ifstream file(path);
  if (!file || !std::filesystem::is_regular_file(path)) {
    throw std::runtime_error("cannot read " + path);
  }
  std::ostringstream text;
  text << file.rdbuf();
  return text.str();
}

// Arguments are LABEL=PATH, baseline first.
std::vector<lg::benchmark::Run> parse_runs(std::span<char*> args) {
  std::vector<lg::benchmark::Run> runs;
  for (const std::string_view arg : args) {
    const auto split = arg.find('=');
    if (split == std::string_view::npos || split == 0) {
      throw std::runtime_error("expected LABEL=PATH: " + std::string(arg));
    }
    runs.push_back(
      {std::string(arg.substr(0, split)), read_file(std::string(arg.substr(split + 1)))}
    );
  }
  return runs;
}

} // namespace

int main(int argc, char** argv) try {
  if (argc < 3) {
    std::fprintf(stderr, "Usage: %s LABEL=run.json LABEL=run.json...\n", argv[0]);
    return 2;
  }
  const auto runs = parse_runs(std::span(argv + 1, static_cast<std::size_t>(argc - 1)));
  lg::benchmark::print_comparison(runs);
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
