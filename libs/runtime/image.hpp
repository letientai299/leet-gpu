#pragma once

#include <cstddef>

struct Image {
  unsigned char *pixels = nullptr;
  unsigned width = 0;
  unsigned height = 0;
  std::size_t size = 0;

  Image() = default;
  Image(const Image &) = delete;
  Image &operator=(const Image &) = delete;
  ~Image();

  std::size_t pixel_count() const;
};

class ImageBytes {
public:
  ImageBytes() = default;
  ImageBytes(const ImageBytes &) = delete;
  ImageBytes &operator=(const ImageBytes &) = delete;
  ~ImageBytes();

  bool upload(const Image &input, std::size_t output_size);
  bool download(Image &output, unsigned width, unsigned height);
  const unsigned char *input() const;
  unsigned char *output();

private:
  unsigned char *input_ = nullptr;
  unsigned char *output_ = nullptr;
  std::size_t output_size_ = 0;
};

using ImageProcessor = bool (*)(const Image &input, Image &output);

int run_image_app(int argc, char **argv, ImageProcessor process);
