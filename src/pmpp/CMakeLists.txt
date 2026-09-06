add_cuda_app(pmpp/ch02/hello)
add_cuda_app(pmpp/ch02/vec-add)
add_cuda_app(pmpp/ch03/grayscale LIBRARIES leet_gpu_image)
add_cuda_app(pmpp/ch03/blur LIBRARIES leet_gpu_image)
add_library(leet_gpu_matmul_kernels STATIC ch03/matmul.kernels.cu)
target_link_libraries(
  leet_gpu_matmul_kernels
  PUBLIC leet_gpu_runtime CCCL::CCCL
  PRIVATE leet_gpu_cutlass
)
add_cuda_app(
  pmpp/ch03/matmul
  EXTENSION cpp
  LIBRARIES leet_gpu_matmul_kernels leet_gpu_nvbench
)
add_cuda_app(
  pmpp/ch03/matmul.bench
  EXTENSION cpp
  LIBRARIES leet_gpu_matmul_kernels leet_gpu_nvbench
)
add_cuda_app(pmpp/ch03/matmul.vec LIBRARIES leet_gpu_matmul_kernels)
