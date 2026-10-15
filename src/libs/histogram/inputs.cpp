#include "histogram/inputs.hpp"

#include <array>
#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <fstream>
#include <spawn.h>
#include <stdexcept>
#include <string_view>
#include <sys/file.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

extern char** environ;

namespace lg::histogram {
namespace {

struct Source {
  const char* url;
  std::size_t size;
  std::uint64_t checksum;
};

// Preserve source bytes and concatenation order.
constexpr std::array kSources{
  Source{"https://www.gutenberg.org/cache/epub/3200/pg3200.txt", 16012562, 9048256224538060916ULL},
  Source{"https://www.gutenberg.org/cache/epub/100/pg100.txt", 5638480, 7686785490689117847ULL},
};

class CacheLock {
public:
  explicit CacheLock(const std::filesystem::path& directory)
      : descriptor_(open((directory / "prepare.lock").c_str(), O_CREAT | O_RDWR, 0600)) {
    if (descriptor_ < 0) {
      throw std::runtime_error("Histogram cache lock open failed");
    }
    if (flock(descriptor_, LOCK_EX) != 0) {
      close(descriptor_);
      throw std::runtime_error("Histogram cache lock failed");
    }
  }
  ~CacheLock() {
    close(descriptor_);
  }
  CacheLock(const CacheLock&) = delete;
  CacheLock& operator=(const CacheLock&) = delete;

private:
  int descriptor_;
};

std::uint64_t checksum(std::string_view bytes) {
  std::uint64_t result = 14695981039346656037ULL;
  for (const unsigned char value : bytes) {
    result = (result ^ value) * 1099511628211ULL;
  }
  return result;
}

void download(const Source& source, const std::filesystem::path& output) {
  std::vector<std::string> args{
    "curl",    "--fail",   "--location",        "--silent", "--show-error",
    "--retry", "2",        "--connect-timeout", "15",       "--max-time",
    "180",     "--output", output.string(),     source.url};
  std::vector<char*> pointers;
  pointers.reserve(args.size() + 1);
  for (auto& arg : args) {
    pointers.push_back(arg.data());
  }
  pointers.push_back(nullptr);
  std::fprintf(stderr, "Downloading %s\n", source.url);
  pid_t child = 0;
  if (posix_spawnp(&child, "curl", nullptr, nullptr, pointers.data(), environ) != 0) {
    throw std::runtime_error("Histogram corpus download requires curl");
  }
  int status = 0;
  while (waitpid(child, &status, 0) < 0) {
    if (errno != EINTR) {
      throw std::runtime_error("Histogram download wait failed");
    }
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    throw std::runtime_error("Histogram corpus download failed");
  }
}

std::string load_bytes(const std::filesystem::path& path, std::size_t size) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file || file.tellg() != static_cast<std::streamoff>(size)) {
    throw std::runtime_error("Histogram corpus size mismatch");
  }
  std::string bytes(size, '\0');
  file.seekg(0);
  if (!file.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
    throw std::runtime_error("Histogram corpus read failed");
  }
  return bytes;
}

void prepare_corpus(const Input& input) {
  std::string corpus;
  corpus.reserve(input.size);
  const auto temporary = input.path.string() + ".part";
  const auto source_path = input.path.string() + ".download";
  try {
    for (const auto& source : kSources) {
      download(source, source_path);
      const auto bytes = load_bytes(source_path, source.size);
      if (checksum(bytes) != source.checksum) {
        throw std::runtime_error("Gutenberg source changed; corpus pin mismatch");
      }
      corpus += bytes;
    }
    if (corpus.size() != input.size || checksum(corpus) != input.checksum) {
      throw std::runtime_error("Histogram corpus checksum mismatch");
    }
    std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
    file.write(corpus.data(), static_cast<std::streamsize>(corpus.size()));
    file.close();
    if (!file) {
      throw std::runtime_error("Histogram corpus write failed");
    }
    std::filesystem::rename(temporary, input.path);
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    std::filesystem::remove(source_path, ignored);
    throw;
  }
  std::filesystem::remove(source_path);
}

} // namespace

std::filesystem::path input_directory() {
  return std::filesystem::path("/tmp") / ("leet-gpu-hist-books-v1-" + std::to_string(getuid()));
}

std::string read_input(const Input& input) {
  auto bytes = load_bytes(input.path, input.size);
  if (checksum(bytes) != input.checksum) {
    throw std::runtime_error("Histogram corpus checksum mismatch");
  }
  return bytes;
}

Input prepare_input(const std::filesystem::path& directory) {
  std::filesystem::create_directories(directory);
  const CacheLock lock(directory);
  const Input input{directory / "gutenberg-english.txt"};
  try {
    read_input(input);
  } catch (const std::runtime_error&) {
    prepare_corpus(input);
  }
  return input;
}

} // namespace lg::histogram
