#include "image.hpp"

#include "checks.hpp"
#include "log.hpp"

#define LODEPNG_NO_COMPILE_CPP
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <lodepng.h>

namespace {

struct ImageArgs {
  const char* input = nullptr;
  const char* output = nullptr;
};

void print_help(const char* program) {
  std::printf("Usage: %s -i <input.png> -o <output.png>\n", program);
}

bool parse_args(int argc, char** argv, ImageArgs& args) {
  for (int index = 1; index < argc; ++index) {
    if (std::strcmp(argv[index], "-i") == 0 && index + 1 < argc) {
      args.input = argv[++index];
      continue;
    }
    if (std::strcmp(argv[index], "-o") == 0 && index + 1 < argc) {
      args.output = argv[++index];
      continue;
    }
    return false;
  }
  return args.input && args.output;
}

bool load_rgb_png(const char* path, Image& image) {
  const unsigned error =
      lodepng_decode_file(&image.pixels, &image.width, &image.height, path, LCT_RGB, 8);
  if (error != 0) {
    HOST_LOG("PNG decode failed: %s", lodepng_error_text(error));
    return false;
  }
  image.size = image.pixel_count() * 3;
  return true;
}

bool save_gray_png(const char* path, const Image& image) {
  const unsigned error =
      lodepng_encode_file(path, image.pixels, image.width, image.height, LCT_GREY, 8);
  if (error != 0) {
    HOST_LOG("PNG encode failed: %s", lodepng_error_text(error));
    return false;
  }
  HOST_LOG("Wrote %ux%u grayscale PNG", image.width, image.height);
  return true;
}

} // namespace

Image::~Image() {
  std::free(pixels);
}

std::size_t Image::pixel_count() const {
  return static_cast<std::size_t>(width) * height;
}

ImageBytes::~ImageBytes() {
  cudaFree(input_);
  cudaFree(output_);
}

bool ImageBytes::upload(const Image& input, std::size_t output_size) {
  output_size_ = output_size;
  if (!CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&input_), input.size)) ||
      !CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&output_), output_size_)) ||
      !CUDA_CHECK(cudaMemcpy(input_, input.pixels, input.size, cudaMemcpyHostToDevice))) {
    return false;
  }
  return true;
}

bool ImageBytes::download(Image& output, unsigned width, unsigned height) {
  output.pixels = static_cast<unsigned char*>(std::malloc(output_size_));
  if (output.pixels == nullptr) {
    HOST_LOG("Image allocation failed: %zu bytes", output_size_);
    return false;
  }
  output.width = width;
  output.height = height;
  output.size = output_size_;
  return CUDA_CHECK(cudaMemcpy(output.pixels, output_, output_size_, cudaMemcpyDeviceToHost));
}

const unsigned char* ImageBytes::input() const {
  return input_;
}

unsigned char* ImageBytes::output() {
  return output_;
}

int run_image_app(int argc, char** argv, ImageProcessor process) {
  if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
    print_help(argv[0]);
    return 0;
  }

  ImageArgs args;
  if (!parse_args(argc, argv, args)) {
    print_help(argv[0]);
    return 2;
  }

  init_log();

  Image input;
  if (!load_rgb_png(args.input, input) || !init_cuda()) {
    return 1;
  }

  Image output;
  if (!process(input, output) || !save_gray_png(args.output, output)) {
    return 1;
  }
  return 0;
}
