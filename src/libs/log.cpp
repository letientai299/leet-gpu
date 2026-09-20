#include "log.hpp"

#include <algorithm>
#include <array>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <string>
#include <string_view>
#include <utility>

namespace {

constexpr std::size_t source_width = 16;
bool color_enabled = false;
std::size_t kind_width = 4;

std::string& gpu_name() {
  static std::string name = "GPU";
  return name;
}

namespace color {

constexpr auto reset = "\033[0m";
constexpr auto time = "\033[2;36m";
constexpr auto source = "\033[2;34m";
constexpr auto host = "\033[1;32m";
constexpr auto gpu = "\033[1;35m";
constexpr auto cuda = "\033[1;31m";

} // namespace color

const char* kind_color(std::string_view kind) {
  if (kind.substr(0, 3) == "sm_") {
    return color::gpu;
  }
  return kind == "CUDA" ? color::cuda : color::host;
}

void append_column(std::string& output, std::string_view value, std::size_t width) {
  output.append(value);
  output.append(value.size() < width ? width - value.size() : 0, ' ');
  output.push_back(' ');
}

void append_colored(
  std::string& output, std::string_view value, std::size_t width, const char* tint
) {
  if (!color_enabled) {
    append_column(output, value, width);
    return;
  }
  output.append(tint);
  append_column(output, value, width);
  output.append(color::reset);
}

std::string format_message(const char* format, va_list args) {
  std::array<char, 512> buffer{};
  va_list retry;
  va_copy(retry, args);
  const int size = std::vsnprintf(buffer.data(), buffer.size(), format, args);
  if (size < 0) {
    va_end(retry);
    return {};
  }
  std::string message(static_cast<std::size_t>(size), '\0');
  if (message.size() < buffer.size()) {
    message.assign(buffer.data(), message.size());
  } else {
    // vsnprintf always writes a terminator, so it needs room for size + 1.
    std::vsnprintf(message.data(), message.size() + 1, format, retry);
  }
  va_end(retry);
  return message;
}

std::string
add_metadata(std::string_view kind, std::string_view file, int line, std::string_view message) {
  const auto slash = file.find_last_of('/');
  const auto basename = slash == std::string_view::npos ? file : file.substr(slash + 1);
  std::string output;
  append_colored(
    output, std::string(basename) + ":" + std::to_string(line), source_width, color::source
  );
  append_colored(output, kind, kind_width, kind_color(kind));
  output.append(message);
  return output;
}

} // namespace

void init_log() {
  if (spdlog::get("cuda") != nullptr) {
    return;
  }
  color_enabled = std::getenv("NO_COLOR") == nullptr;
  auto logger = spdlog::stdout_color_mt("cuda");
  const std::string stamp =
    color_enabled ? std::string(color::time) + "%T.%e" + color::reset : std::string("%T.%e");
  logger->set_pattern(stamp + " %v");
  spdlog::set_default_logger(std::move(logger));
}

void set_gpu_arch(int major, int minor) {
  gpu_name() = "sm_" + std::to_string(major) + std::to_string(minor);
  kind_width = std::max(kind_width, gpu_name().size());
}

const char* gpu_arch() {
  return gpu_name().c_str();
}

void write_log(const char* kind, const char* file, int line, const char* format, ...) {
  va_list args;
  va_start(args, format);
  const auto message = format_message(format, args);
  va_end(args);
  spdlog::default_logger_raw()->log(
    {file, line, ""}, spdlog::level::info, add_metadata(kind, file, line, message)
  );
}
