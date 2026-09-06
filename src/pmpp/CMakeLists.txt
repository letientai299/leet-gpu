add_cuda_app(pmpp/ch02/hello)
add_cuda_app(pmpp/ch02/vec-add)
add_cuda_app(pmpp/ch03/grayscale LIBRARIES lg_image)
add_cuda_app(pmpp/ch03/blur LIBRARIES lg_image)
add_library(lg_matmul_kernels STATIC ch03/matmul.kernels.cu)
target_link_libraries(
  lg_matmul_kernels
  PUBLIC lg_runtime CCCL::CCCL
  PRIVATE lg_cutlass
)
add_cuda_app(pmpp/ch03/matmul LIBRARIES lg_matmul_kernels lg_nvbench)
add_cuda_app(pmpp/ch03/matmul.bench LIBRARIES lg_matmul_kernels lg_nvbench)
add_cuda_app(pmpp/ch03/matmul.vec LIBRARIES lg_matmul_kernels)
