add_cuda_app(pmpp/ch02/hello)
add_cuda_app(pmpp/ch02/vec-add)
add_cuda_app(pmpp/ch03/grayscale LIBRARIES lg_image)
add_cuda_app(pmpp/ch03/blur LIBRARIES lg_image)
add_library(lg_matmul_kernels STATIC ch03/matmul.kernels.cu)
target_link_libraries(lg_matmul_kernels PUBLIC lg_matmul)
add_cuda_app(pmpp/ch03/matmul LIBRARIES lg_matmul_kernels lg_matmul_nvbench)
add_cuda_app(
  pmpp/ch03/matmul.bench
  LIBRARIES lg_matmul_kernels lg_matmul_nvbench
)
add_cuda_app(pmpp/ch03/matmul.vec LIBRARIES lg_matmul_kernels)
add_cuda_app(pmpp/ch04/limits LIBRARIES lg_cuda_runtime)
add_cuda_app(pmpp/ch07/conv.basic LIBRARIES lg_convolution_nvbench)
add_cuda_app(pmpp/ch07/conv.constant LIBRARIES lg_convolution_nvbench)
add_cuda_app(pmpp/ch07/conv.tiled LIBRARIES lg_convolution_nvbench)
add_cuda_app(pmpp/ch07/conv.cached LIBRARIES lg_convolution_nvbench)
add_cuda_app(pmpp/ch07/conv3d.basic LIBRARIES lg_convolution3d_nvbench)
add_cuda_app(pmpp/ch07/conv3d.constant LIBRARIES lg_convolution3d_nvbench)
add_cuda_app(pmpp/ch07/conv3d.tiled LIBRARIES lg_convolution3d_nvbench)
add_cuda_app(
  pmpp/ch05/matmul.tile
  LIBRARIES lg_matmul_kernels lg_matmul_nvbench
)
set_source_files_properties(
  "${CMAKE_CURRENT_SOURCE_DIR}/ch05/matmul.tile.cpp"
  PROPERTIES LANGUAGE CUDA
)
add_cuda_app(
  pmpp/ch06/matmul.corner
  LIBRARIES lg_matmul_kernels lg_matmul_nvbench
)
set_source_files_properties(
  "${CMAKE_CURRENT_SOURCE_DIR}/ch06/matmul.corner.cpp"
  PROPERTIES LANGUAGE CUDA
)
add_cuda_app(
  pmpp/ch06/matmul.coarse
  LIBRARIES lg_matmul_kernels lg_matmul_nvbench
)
set_source_files_properties(
  "${CMAKE_CURRENT_SOURCE_DIR}/ch06/matmul.coarse.cpp"
  PROPERTIES LANGUAGE CUDA
)
