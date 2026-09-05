#include "image.hpp"

#include <cstddef>
#include <cuda/cmath>

/// Convert RGB pixels in parallel.
__global__ void grayscale_kernel(const ImageByte* input,
                                 ImageByte* output,
                                 const unsigned width,
                                 const unsigned height) {
  const auto pos = image_thread();
  if (!pos.in_bounds(width, height)) {
    return;
  }

  const std::size_t pixel = static_cast<std::size_t>(pos.row) * width + pos.column;
  const std::size_t rgb = pixel * 3;
  const auto r = static_cast<float>(input[rgb]);
  const auto g = static_cast<float>(input[rgb + 1]);
  const auto b = static_cast<float>(input[rgb + 2]);
  output[pixel] = static_cast<ImageByte>(0.21F * r + 0.72F * g + 0.07F * b);
}

static bool grayscale_on_gpu(const Image& rgb, Image& gray) {
  ImageBytes bytes;
  if (!bytes.upload(rgb, rgb.pixel_count())) {
    return false;
  }

  constexpr dim3 block(16, 16);
  const dim3 grid(cuda::ceil_div(rgb.width, block.x), cuda::ceil_div(rgb.height, block.y));
  log_image_launch(grid, block, rgb.pixel_count());
  grayscale_kernel<<<grid, block>>>(bytes.input(), bytes.output(), rgb.width, rgb.height);

  return CUDA_CHECK(cudaGetLastError()) && bytes.download(gray, rgb.width, rgb.height);
}

int main(int argc, char** argv) {
  return run_image_app(argc, argv, grayscale_on_gpu);
}
