#include "histogram/inputs.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace {

void require(bool condition) {
  if (!condition) {
    throw std::runtime_error("Histogram cache regression failed");
  }
}

class Directory {
public:
  Directory() {
    std::string pattern = "/tmp/leet-gpu-hist-test-" + std::to_string(getuid()) + "-XXXXXX";
    if (mkdtemp(pattern.data()) == nullptr) {
      throw std::runtime_error("Histogram test directory failed");
    }
    path = pattern;
  }
  ~Directory() {
    std::filesystem::remove_all(path);
  }
  Directory(const Directory&) = delete;
  Directory& operator=(const Directory&) = delete;

  std::filesystem::path path;
};

void check_counts(const lg::histogram::Input& input) {
  const auto bytes = lg::histogram::read_input(input);
  require(bytes.size() > std::size_t{20} * 1024 * 1024);
  std::array<unsigned int, 7> bins{};
  for (const char value : bytes) {
    if (value >= 'a' && value <= 'z') {
      ++bins[(value - 'a') / 4];
    }
  }
  require(bins == input.bins);
}

void check_recovery(const std::filesystem::path& directory, const lg::histogram::Input& source) {
  const auto path = directory / "gutenberg-english.txt";
  std::filesystem::copy_file(source.path, path);
  const auto modified = std::filesystem::last_write_time(path);
  const auto input = lg::histogram::prepare_input(directory);
  require(std::filesystem::last_write_time(path) == modified);
  {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    file.put('!');
  }
  bool rejected = false;
  try {
    lg::histogram::read_input(input);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected);
  const auto recovered = lg::histogram::prepare_input(directory);
  require(lg::histogram::read_input(recovered) == lg::histogram::read_input(source));
  require(!std::filesystem::exists(path.string() + ".part"));
  require(!std::filesystem::exists(path.string() + ".download"));
  check_counts(recovered);
}

void check_concurrent(const std::filesystem::path& directory) {
  const auto child = fork();
  require(child >= 0);
  if (child == 0) {
    try {
      lg::histogram::prepare_input(directory);
      _exit(0);
    } catch (...) {
      _exit(1);
    }
  }
  const auto input = lg::histogram::prepare_input(directory);
  int status = 0;
  require(waitpid(child, &status, 0) == child);
  require(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  check_counts(input);
}

} // namespace

int main() try {
  const auto source = lg::histogram::prepare_input(lg::histogram::input_directory());
  check_counts(source);
  const Directory directory;
  check_recovery(directory.path, source);
  check_concurrent(directory.path);
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
