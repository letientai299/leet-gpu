#pragma once

#include <cstdio>
#include <fmt/format.h>
#include <string_view>
#include <utility>

void init_log();
void set_gpu_arch(int major, int minor);
const char* gpu_arch();
/// True when `stream` is a terminal or FORCE_COLOR is set, and NO_COLOR is unset.
bool use_color(std::FILE* stream);

void write_log_message(const char* kind, const char* file, int line, std::string_view message);

template <typename... Args>
void write_log(
  const char* kind, const char* file, int line, fmt::format_string<Args...> format, Args&&... args
) {
  write_log_message(kind, file, line, fmt::format(format, std::forward<Args>(args)...));
}

#define HOST_LOG(...) write_log("HOST", __FILE__, __LINE__, __VA_ARGS__)
#define GPU_LOG(...) write_log(gpu_arch(), __FILE__, __LINE__, __VA_ARGS__)
