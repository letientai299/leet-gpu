#include "cuda_check.hpp"
#include "image.hpp"

#include <cstdlib>

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
