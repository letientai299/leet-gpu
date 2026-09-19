#pragma once

void init_log();
void set_gpu_arch(int major, int minor);
const char* gpu_arch();

// printf attribute makes the compiler type-check the varargs.
// https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html
#if defined(__GNUC__) || defined(__clang__)
#define LG_PRINTF_FORMAT(fmt, first) __attribute__((format(printf, fmt, first)))
#else
#define LG_PRINTF_FORMAT(fmt, first)
#endif

LG_PRINTF_FORMAT(4, 5)
void write_log(const char* kind, const char* file, int line, const char* format, ...);

#define HOST_LOG(...) write_log("HOST", __FILE__, __LINE__, __VA_ARGS__)
#define GPU_LOG(...) write_log(gpu_arch(), __FILE__, __LINE__, __VA_ARGS__)
