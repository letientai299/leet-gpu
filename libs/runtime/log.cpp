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

namespace {

constexpr size_t source_width = 16;
bool color_enabled = false;
size_t kind_width = 4;

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

const char* kind_color(const char* kind) {
  if (std::string_view(kind).substr(0, 3) == "sm_") {
    return color::gpu;
  }
  if (strcmp(kind, "CUDA") == 0) {
    return color::cuda;
  }
  return color::host;
}

void append_column(std::string& output, std::string_view value, size_t width) {
  output.append(value);
  if (value.size() < width) {
    output.append(width - value.size(), ' ');
  }
  output.push_back(' ');
}

std::string format_message(const char* format, va_list args) {
  std::array<char, 512> buffer{};
  va_list copy;
  va_copy(copy, args);
  const int size = vsnprintf(buffer.data(), buffer.size(), format, copy);
  va_end(copy);
  if (size < 0) {
    return {};
  }
  if (static_cast<size_t>(size) < buffer.size()) {
    return buffer.data();
  }

  std::string message(static_cast<size_t>(size) + 1, '\0');
  vsnprintf(message.data(), message.size(), format, args);
  message.resize(static_cast<size_t>(size));
  return message;
}

std::string add_metadata(const char* kind, const char* file, int line, const std::string& message) {
  const char* basename = strrchr(file, '/');
  basename = basename ? basename + 1 : file;
  const std::string source = std::string(basename) + ":" + std::to_string(line);
  std::string output;
  if (color_enabled) {
    output.append(color::source);
  }
  append_column(output, source, source_width);
  if (color_enabled) {
    output.append(color::reset);
    output.append(kind_color(kind));
  }
  append_column(output, kind, kind_width);
  if (color_enabled) {
    output.append(color::reset);
  }
  output.append(message);
  return output;
}

} // namespace

void init_log() {
  color_enabled = std::getenv("NO_COLOR") == nullptr;
  auto logger = spdlog::stdout_color_mt("cuda");
  if (color_enabled) {
    logger->set_pattern(std::string(color::time) + "%T.%e" + color::reset + " %v");
  } else {
    logger->set_pattern("%T.%e %v");
  }
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
  spdlog::default_logger_raw()->log({file, line, ""}, spdlog::level::info,
                                    add_metadata(kind, file, line, message));
}
