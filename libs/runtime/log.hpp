#pragma once

void init_log();
void set_gpu_arch(int major, int minor);
const char* gpu_arch();
void write_log(const char* kind, const char* file, int line, const char* format, ...);

#define HOST_LOG(...) write_log("HOST", __FILE__, __LINE__, __VA_ARGS__)
#define GPU_LOG(...) write_log(gpu_arch(), __FILE__, __LINE__, __VA_ARGS__)
