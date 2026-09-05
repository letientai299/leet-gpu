#include "cuda_check.hpp"

#define LODEPNG_NO_COMPILE_CPP
#include <lodepng.h>

#include <cstdio>
#include <cstring>

namespace {
struct Options {
  const char *input = nullptr;
  const char *output = nullptr;
};

struct Image {
  unsigned char *pixels = nullptr;
  unsigned width = 0;
  unsigned height = 0;

  Image() = default;
  Image(const Image &) = delete;
  Image &operator=(const Image &) = delete;
  ~Image() { std::free(pixels); }
};
} // namespace

/// Convert RGB pixels in parallel.
__global__ static void grayscale_kernel(const unsigned char *input,
                                        unsigned char *output, unsigned width,
                                        unsigned height) {
  const unsigned column = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
  if (column >= width || row >= height) {
    return;
  }

  const size_t pixel = static_cast<size_t>(row) * width + column;
  const size_t rgb = pixel * 3;
  output[pixel] =
      static_cast<unsigned char>(0.21F * static_cast<float>(input[rgb]) +
                                 0.72F * static_cast<float>(input[rgb + 1]) +
                                 0.07F * static_cast<float>(input[rgb + 2]));
}

static void print_help(const char *program) {
  std::printf("Usage: %s -i <input.png> -o <output.png>\n", program);
}

static bool parse_args(int argc, char **argv, Options &options) {
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
static bool load_rgb_png(const char *path, Image &image) {
  const unsigned error = lodepng_decode_file(&image.pixels, &image.width,
                                             &image.height, path, LCT_RGB, 8);
  if (error != 0) {
    HOST_LOG("PNG decode failed: %s", lodepng_error_text(error));
    return false;
  }
  return true;
}

/// Convert RGB pixels on the GPU.
static bool grayscale_on_gpu(const Image &rgb, Image &gray) {
  const size_t pixels = static_cast<size_t>(rgb.width) * rgb.height;
  const size_t rgb_bytes = pixels * 3;
  unsigned char *device_input = nullptr;
  unsigned char *device_output = nullptr;
  if (!CUDA_CHECK(cudaMalloc(&device_input, rgb_bytes)) ||
      !CUDA_CHECK(cudaMalloc(&device_output, pixels)) ||
      !CUDA_CHECK(cudaMemcpy(device_input, rgb.pixels, rgb_bytes,
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

  auto *host_gray = static_cast<unsigned char *>(std::malloc(pixels));
  if (host_gray == nullptr) {
    cudaFree(device_input);
    cudaFree(device_output);
    HOST_LOG("Failed to allocate %zu grayscale bytes", pixels);
    return false;
  }
  gray.pixels = host_gray;
  gray.width = rgb.width;
  gray.height = rgb.height;
  const bool ok = CUDA_CHECK(cudaGetLastError()) &&
                  CUDA_CHECK(cudaMemcpy(gray.pixels, device_output, pixels,
                                        cudaMemcpyDeviceToHost));
  cudaFree(device_input);
  cudaFree(device_output);
  return ok;
}

/// Write a grayscale PNG.
static bool save_gray_png(const char *path, const Image &image) {
  const unsigned error = lodepng_encode_file(path, image.pixels, image.width,
                                             image.height, LCT_GREY, 8);
  if (error != 0) {
    HOST_LOG("PNG encode failed: %s", lodepng_error_text(error));
    return false;
  }
  HOST_LOG("Wrote %ux%u grayscale PNG", image.width, image.height);
  return true;
}

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
