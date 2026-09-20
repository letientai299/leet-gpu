#include "image.hpp"

#include "checks.hpp"
#include "log.hpp"

#define LODEPNG_NO_COMPILE_CPP
#include <array>
#include <cstdio>
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
  for (int index = 1; index + 1 < argc; index += 2) {
    const char* flag = argv[index];
    if (std::strcmp(flag, "-i") == 0) {
      args.input = argv[index + 1];
    } else if (std::strcmp(flag, "-o") == 0) {
      args.output = argv[index + 1];
    } else {
      return false;
    }
  }
  return args.input != nullptr && args.output != nullptr;
}

/// Channel count maps 1:1 onto a lodepng grey/RGB color type.
constexpr std::array<LodePNGColorType, 4> kColorTypes{LCT_GREY, LCT_GREY_ALPHA, LCT_RGB, LCT_RGBA};

bool load_rgb_png(const char* path, Image& image) {
  ImageByte* decoded = nullptr;
  const unsigned error =
    lodepng_decode_file(&decoded, &image.width, &image.height, path, LCT_RGB, 8);
  image.pixels.reset(decoded);
  if (error != 0) {
    HOST_LOG("PNG decode failed: %s", lodepng_error_text(error));
    return false;
  }
  image.size = image.pixel_count() * 3;
  return true;
}

bool save_png(const char* path, const Image& image) {
  const auto channels = image.channels();
  if (channels == 0 || channels > kColorTypes.size() ||
      channels * image.pixel_count() != image.size) {
    HOST_LOG(
      "PNG encode failed: invalid size %zu for %ux%u", image.size, image.width, image.height
    );
    return false;
  }

  const unsigned error = lodepng_encode_file(
    path, image.pixels.get(), image.width, image.height, kColorTypes[channels - 1], 8
  );
  if (error != 0) {
    HOST_LOG("PNG encode failed: %s", lodepng_error_text(error));
    return false;
  }
  HOST_LOG("Wrote %ux%u %s", image.width, image.height, path);
  return true;
}

} // namespace

bool ImageBytes::upload(const Image& input, std::size_t output_size) {
  output_size_ = output_size;
  input_ =
    cuda::device_buffer<ImageByte>{default_stream(), device_pool(), input.size, cuda::no_init};
  output_ =
    cuda::device_buffer<ImageByte>{default_stream(), device_pool(), output_size_, cuda::no_init};
  return CUDA_CHECK(
    cudaMemcpy(input_.data(), input.pixels.get(), input.size, cudaMemcpyHostToDevice)
  );
}

bool ImageBytes::download(Image& output, unsigned width, unsigned height) const {
  output.pixels.reset(static_cast<ImageByte*>(std::malloc(output_size_)));
  if (output.pixels == nullptr) {
    HOST_LOG("Image allocation failed: %zu bytes", output_size_);
    return false;
  }
  output.width = width;
  output.height = height;
  output.size = output_size_;
  auto destination = output.bytes();
  return COPY_CHECK(output_, destination);
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

  init_log();
  Image input;
  if (!load_rgb_png(args.input, input) || !init_cuda()) {
    return 1;
  }

  Image output;
  return process(input, output) && save_png(args.output, output) ? 0 : 1;
}
