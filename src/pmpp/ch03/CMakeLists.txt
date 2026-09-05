add_cuda_app(
  KEY pmpp/ch03/grayscale
  SOURCE grayscale.cu
  LIBRARIES leet_gpu_image
)

add_cuda_app(KEY pmpp/ch03/blur SOURCE blur.cu LIBRARIES leet_gpu_image)
