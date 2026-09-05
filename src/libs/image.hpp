#pragma once

#include "device.hpp"

#include <cstddef>

using ImageByte = unsigned char;

struct Image {
  ImageByte* pixels = nullptr;
  unsigned width = 0;
  unsigned height = 0;
  std::size_t size = 0;

  Image() = default;
  Image(const Image&) = delete;
  Image& operator=(const Image&) = delete;
  ~Image();

  [[nodiscard]] std::size_t pixel_count() const;
};

class ImageBytes {
public:
  bool upload(const Image& input, std::size_t output_size);
  bool download(Image& output, unsigned width, unsigned height) const;
  [[nodiscard]] const ImageByte* input() const;
  [[nodiscard]] ImageByte* output() const;

private:
  DeviceBuffer<ImageByte> input_;
  DeviceBuffer<ImageByte> output_;
  std::size_t output_size_ = 0;
};

using ImageProcessor = bool (*)(const Image& input, Image& output);

int run_image_app(int argc, char** argv, ImageProcessor process);
