#pragma once

#include "checks.hpp"

#include <cstddef>
#include <cstdlib>
#include <cuda/buffer>
#include <cuda/std/span>
#include <memory>

using ImageByte = unsigned char;

/// lodepng hands back malloc'd pixels, so the deleter must be free.
/// https://en.cppreference.com/w/cpp/memory/unique_ptr
struct FreeDeleter {
  void operator()(ImageByte* pointer) const {
    std::free(pointer);
  }
};

using ImageBuffer = std::unique_ptr<ImageByte[], FreeDeleter>;

struct Image {
  ImageBuffer pixels;
  unsigned width = 0;
  unsigned height = 0;
  std::size_t size = 0;

  [[nodiscard]] std::size_t pixel_count() const {
    return static_cast<std::size_t>(width) * height;
  }
  [[nodiscard]] std::size_t channels() const {
    const auto pixels_count = pixel_count();
    return pixels_count == 0 ? 0 : size / pixels_count;
  }
  [[nodiscard]] cuda::std::span<ImageByte> bytes() const {
    return {pixels.get(), size};
  }
};

class ImageBytes {
public:
  bool upload(const Image& input, std::size_t output_size);
  bool download(Image& output, unsigned width, unsigned height) const;
  [[nodiscard]] const ImageByte* input() const {
    return input_.data();
  }
  [[nodiscard]] ImageByte* output() {
    return output_.data();
  }

private:
  cuda::device_buffer<ImageByte> input_{default_stream(), device_pool()};
  cuda::device_buffer<ImageByte> output_{default_stream(), device_pool()};
  std::size_t output_size_ = 0;
};

using ImageProcessor = bool (*)(const Image& input, Image& output);

int run_image_app(int argc, char** argv, ImageProcessor process);

#ifdef __CUDACC__
struct ImageThread {
  unsigned column;
  unsigned row;

  [[nodiscard]] __device__ bool in_bounds(unsigned width, unsigned height) const {
    return column < width && row < height;
  }
};

__device__ inline ImageThread image_thread() {
  return {
    blockIdx.x * blockDim.x + threadIdx.x,
    blockIdx.y * blockDim.y + threadIdx.y,
  };
}
#endif
