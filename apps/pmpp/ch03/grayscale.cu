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

struct Image {
  std::vector<unsigned char> pixels;
  unsigned width = 0;
  unsigned height = 0;
};

/// Convert RGB pixels in parallel.
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

/// Load an RGB PNG.
bool load_rgb_png(const char *path, Image &image) {
  const unsigned error = lodepng::decode(
      image.pixels, image.width, image.height, std::string(path), LCT_RGB, 8);
  if (error != 0) {
    HOST_LOG("PNG decode failed: %s", lodepng_error_text(error));
    return false;
  }
  return true;
}

/// Convert RGB pixels on the GPU.
bool grayscale_on_gpu(const Image &rgb, Image &gray) {
  const size_t pixels = static_cast<size_t>(rgb.width) * rgb.height;
  unsigned char *device_input = nullptr;
  unsigned char *device_output = nullptr;
  if (!CUDA_CHECK(cudaMalloc(&device_input, rgb.pixels.size())) ||
      !CUDA_CHECK(cudaMalloc(&device_output, pixels)) ||
      !CUDA_CHECK(cudaMemcpy(device_input, rgb.pixels.data(), rgb.pixels.size(),
                             cudaMemcpyHostToDevice))) {
    cudaFree(device_input);
    cudaFree(device_output);
    return false;
  }

  constexpr dim3 block(16, 16);
  const dim3 grid((rgb.width + block.x - 1) / block.x,
                  (rgb.height + block.y - 1) / block.y);
  grayscale_kernel<<<grid, block>>>(device_input, device_output, rgb.width,
                                    rgb.height);

  gray.pixels.resize(pixels);
  gray.width = rgb.width;
  gray.height = rgb.height;
  const bool ok = CUDA_CHECK(cudaGetLastError()) &&
                  CUDA_CHECK(cudaMemcpy(gray.pixels.data(), device_output,
                                        pixels, cudaMemcpyDeviceToHost));
  cudaFree(device_input);
  cudaFree(device_output);
  return ok;
}

/// Write a grayscale PNG.
bool save_gray_png(const char *path, const Image &image) {
  const unsigned error = lodepng::encode(
      std::string(path), image.pixels, image.width, image.height, LCT_GREY, 8);
  if (error != 0) {
    HOST_LOG("PNG encode failed: %s", lodepng_error_text(error));
    return false;
  }
  HOST_LOG("Wrote %ux%u grayscale PNG", image.width, image.height);
  return true;
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

  Image rgb;
  if (!load_rgb_png(options.input, rgb) || !init_cuda()) {
    return 1;
  }

  Image gray;
  if (!grayscale_on_gpu(rgb, gray) || !save_gray_png(options.output, gray)) {
    return 1;
  }
  return 0;
}
