#include "checks.hpp"
#include "image.hpp"

#define BLUR_RADIUS 7U

__global__ static void blur_kernel(const ImageByte* input,
                                   ImageByte* output,
                                   const unsigned width,
                                   const unsigned height) {
  const unsigned column = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned row = blockIdx.y * blockDim.y + threadIdx.y;

  if (column >= width || row >= height) {
    return; // the location is not exists in the image.
  }

  const unsigned y0 = row - min(row, BLUR_RADIUS);
  const unsigned y1 = min(row + BLUR_RADIUS, height - 1);
  const unsigned x0 = column - min(column, BLUR_RADIUS);
  const unsigned x1 = min(column + BLUR_RADIUS, width - 1);
  const int total = static_cast<int>((y1 - y0 + 1) * (x1 - x0 + 1));

  int sum[3] = {}; // RGB
  for (unsigned y = y0; y <= y1; y++) {
    for (unsigned x = x0; x <= x1; x++) {
      const auto p = (static_cast<size_t>(y) * width + x) * 3;
      for (int channel = 0; channel < 3; channel++) {
        sum[channel] += input[p + channel];
      }
    }
  }

  const auto pixel = static_cast<size_t>(row) * width + column;
  for (int c = 0; c < 3; c++) {
    output[pixel * 3 + c] = static_cast<ImageByte>(sum[c] / total);
  }
}

static bool blur_on_gpu(const Image& rgb, Image& blur) {
  ImageBytes bytes;
  if (!bytes.upload(rgb, rgb.size)) {
    return false;
  }

  constexpr dim3 block(16, 16);
  const dim3 grid((rgb.width + block.x - 1) / block.x, (rgb.height + block.y - 1) / block.y);
  blur_kernel<<<grid, block>>>(bytes.input(), bytes.output(), rgb.width, rgb.height);

  return CUDA_CHECK(cudaGetLastError()) && bytes.download(blur, rgb.width, rgb.height);
}

int main(int argc, char** argv) {
  return run_image_app(argc, argv, blur_on_gpu);
}
