#include "checks.hpp"
#include "image.hpp"

/// Convert RGB pixels in parallel.
__global__ static void
grayscale_kernel(const ImageByte* input, ImageByte* output, unsigned width, unsigned height) {
  const unsigned column = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
  if (column >= width || row >= height) {
    return;
  }

  const size_t pixel = static_cast<size_t>(row) * width + column;
  const size_t rgb = pixel * 3;
  auto r = static_cast<float>(input[rgb]);
  auto g = static_cast<float>(input[rgb + 1]);
  auto b = static_cast<float>(input[rgb + 2]);
  output[pixel] = static_cast<ImageByte>(0.21F * r + 0.72F * g + 0.07F * b);
}

static bool grayscale_on_gpu(const Image& rgb, Image& gray) {
  ImageBytes bytes;
  if (!bytes.upload(rgb, rgb.pixel_count())) {
    return false;
  }

  constexpr dim3 block(16, 16);
  const dim3 grid((rgb.width + block.x - 1) / block.x, (rgb.height + block.y - 1) / block.y);
  grayscale_kernel<<<grid, block>>>(bytes.input(), bytes.output(), rgb.width, rgb.height);

  return CUDA_CHECK(cudaGetLastError()) && bytes.download(gray, rgb.width, rgb.height);
}

int main(int argc, char** argv) {
  return run_image_app(argc, argv, grayscale_on_gpu);
}
