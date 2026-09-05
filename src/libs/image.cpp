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

bool save_png(const char* path, const Image& image) {
  const auto pixels = image.pixel_count();
  if (pixels == 0 || image.size % pixels != 0) {
    HOST_LOG("PNG encode failed: invalid size %zu for %ux%u", image.size, image.width,
             image.height);
    return false;
  }

  const auto channels = image.size / pixels;
  LodePNGColorType color_type = LCT_GREY;
  switch (channels) {
  case 1:
    color_type = LCT_GREY;
    break;
  case 2:
    color_type = LCT_GREY_ALPHA;
    break;
  case 3:
    color_type = LCT_RGB;
    break;
  case 4:
    color_type = LCT_RGBA;
    break;
  default:
    HOST_LOG("PNG encode failed: unsupported channel count %zu", channels);
    return false;
  }

  const unsigned error =
      lodepng_encode_file(path, image.pixels, image.width, image.height, color_type, 8);
  if (error != 0) {
    HOST_LOG("PNG encode failed: %s", lodepng_error_text(error));
    return false;
  }
  HOST_LOG("Wrote %ux%u %s", image.width, image.height, path);
  return true;
}

} // namespace

Image::~Image() {
  std::free(pixels);
}

std::size_t Image::pixel_count() const {
  return static_cast<std::size_t>(width) * height;
}

bool ImageBytes::upload(const Image& input, std::size_t output_size) {
  output_size_ = output_size;
  return input_.allocate(input.size) && output_.allocate(output_size_) &&
         CUDA_CHECK(cudaMemcpy(input_.get(), input.pixels, input.size, cudaMemcpyHostToDevice));
}

bool ImageBytes::download(Image& output, unsigned width, unsigned height) const {
  output.pixels = static_cast<ImageByte*>(std::malloc(output_size_));
  if (output.pixels == nullptr) {
    HOST_LOG("Image allocation failed: %zu bytes", output_size_);
    return false;
  }
  output.width = width;
  output.height = height;
  output.size = output_size_;
  return CUDA_CHECK(cudaMemcpy(output.pixels, output_.get(), output_size_, cudaMemcpyDeviceToHost));
}

const ImageByte* ImageBytes::input() const {
  return input_.get();
}

ImageByte* ImageBytes::output() const {
  return output_.get();
}

int run_image_app(int argc, char** argv, ImageProcessor process) {
  if (help_requested(argc, argv)) {
    print_help(argv[0]);
    return 0;
  }

  ImageArgs args;
  if (!parse_args(argc, argv, args)) {
    print_help(argv[0]);
    return 2;
  }

  Image input;
  init_log();
  if (!load_rgb_png(args.input, input) || !init_cuda()) {
    return 1;
  }

  Image output;
  if (!process(input, output) || !save_png(args.output, output)) {
    return 1;
  }
  return 0;
}
