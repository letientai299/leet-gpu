#include "cuda_check.hpp"

#include <lodepng.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Options {
  const char *input = nullptr;
  const char *output = nullptr;
};

__global__ void grayscale_kernel(const unsigned char *input,
                                 unsigned char *output, unsigned width,
                                 unsigned height) {
  const unsigned column = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
  if (column >= width || row >= height) {
    return;
  }

  const size_t pixel = static_cast<size_t>(row) * width + column;
  const size_t rgb = pixel * 3;
  output[pixel] = static_cast<unsigned char>(
      0.21F * input[rgb] + 0.72F * input[rgb + 1] + 0.07F * input[rgb + 2]);
}

void print_help(const char *program) {
  std::printf("Usage: %s -i <input.png> -o <output.png>\n", program);
}

bool parse_args(int argc, char **argv, Options &options) {
  for (int index = 1; index < argc; ++index) {
    if (std::strcmp(argv[index], "-i") == 0 && index + 1 < argc) {
      options.input = argv[++index];
      continue;
    }
    if (std::strcmp(argv[index], "-o") == 0 && index + 1 < argc) {
      options.output = argv[++index];
      continue;
    }
    return false;
  }
  return options.input && options.output;
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
    print_help(argv[0]);
    return 0;
  }

  Options options;
  if (!parse_args(argc, argv, options)) {
    print_help(argv[0]);
    return 2;
  }

  init_log();

  std::vector<unsigned char> input;
  unsigned width = 0;
  unsigned height = 0;
  unsigned error = lodepng::decode(input, width, height,
                                   std::string(options.input), LCT_RGB, 8);
  if (error != 0) {
    HOST_LOG("PNG decode failed: %s", lodepng_error_text(error));
    return 1;
  }
  if (!init_cuda()) {
    return 1;
  }

  const size_t pixels = static_cast<size_t>(width) * height;
  unsigned char *device_input = nullptr;
  unsigned char *device_output = nullptr;
  if (!CUDA_CHECK(cudaMalloc(&device_input, input.size())) ||
      !CUDA_CHECK(cudaMalloc(&device_output, pixels)) ||
      !CUDA_CHECK(cudaMemcpy(device_input, input.data(), input.size(),
                             cudaMemcpyHostToDevice))) {
    cudaFree(device_input);
    cudaFree(device_output);
    return 1;
  }

  constexpr dim3 block(16, 16);
  const dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);
  grayscale_kernel<<<grid, block>>>(device_input, device_output, width, height);

  std::vector<unsigned char> output(pixels);
  const bool ok = CUDA_CHECK(cudaGetLastError()) &&
                  CUDA_CHECK(cudaMemcpy(output.data(), device_output, pixels,
                                        cudaMemcpyDeviceToHost));
  cudaFree(device_input);
  cudaFree(device_output);
  if (!ok) {
    return 1;
  }

  error = lodepng::encode(std::string(options.output), output, width, height,
                          LCT_GREY, 8);
  if (error != 0) {
    HOST_LOG("PNG encode failed: %s", lodepng_error_text(error));
    return 1;
  }

  HOST_LOG("Wrote %ux%u grayscale PNG", width, height);
  return 0;
}
