#include "image.hpp"

#include <cstddef>
#include <cuda/cmath>

constexpr unsigned kBlurRadius = 7;

__global__ void blur_kernel(const ImageByte* input,
                            ImageByte* output,
                            const unsigned width,
                            const unsigned height) {
  const auto pos = image_thread();
  if (!pos.in_bounds(width, height)) {
    return;
  }

  const unsigned y0 = pos.row - min(pos.row, kBlurRadius);
  const unsigned y1 = min(pos.row + kBlurRadius, height - 1);
  const unsigned x0 = pos.column - min(pos.column, kBlurRadius);
  const unsigned x1 = min(pos.column + kBlurRadius, width - 1);
  const int total = static_cast<int>((y1 - y0 + 1) * (x1 - x0 + 1));

  int sum[3] = {};
  for (unsigned y = y0; y <= y1; ++y) {
    for (unsigned x = x0; x <= x1; ++x) {
      const auto p = (static_cast<std::size_t>(y) * width + x) * 3;
      for (int channel = 0; channel < 3; ++channel) {
        sum[channel] += input[p + channel];
      }
    }
  }

  const auto pixel = static_cast<std::size_t>(pos.row) * width + pos.column;
  for (int c = 0; c < 3; ++c) {
    output[pixel * 3 + c] = static_cast<ImageByte>(sum[c] / total);
  }
}

static bool blur_on_gpu(const Image& rgb, Image& blur) {
  ImageBytes bytes;
  if (!bytes.upload(rgb, rgb.size)) {
    return false;
  }

  constexpr dim3 block(16, 16);
  const dim3 grid(cuda::ceil_div(rgb.width, block.x), cuda::ceil_div(rgb.height, block.y));
  log_image_launch(grid, block, rgb.pixel_count());
  blur_kernel<<<grid, block>>>(bytes.input(), bytes.output(), rgb.width, rgb.height);

  return CUDA_CHECK(cudaGetLastError()) && bytes.download(blur, rgb.width, rgb.height);
}

int main(int argc, char** argv) {
  return run_image_app(argc, argv, blur_on_gpu);
}
