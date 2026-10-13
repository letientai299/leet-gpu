set(LG_DEP_CACHE "${CMAKE_SOURCE_DIR}/.cache/fetchcontent")

# Keep extracted sources outside build/ so wiping the CMake tree skips
# re-download. Binary dirs stay per-prefix (build/docker vs build/local).
function(lg_fetch name)
  cmake_parse_arguments(DEP "" "URL;HASH" "" ${ARGN})
  if(NOT DEP_URL OR NOT DEP_HASH OR DEP_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "lg_fetch(${name}) needs URL and HASH")
  endif()
  FetchContent_Declare(
    ${name}
    URL "${DEP_URL}"
    URL_HASH "SHA256=${DEP_HASH}"
    SOURCE_DIR "${LG_DEP_CACHE}/${name}-src"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
endfunction()

lg_fetch(
  spdlog
  URL https://github.com/gabime/spdlog/archive/refs/tags/v1.16.0.tar.gz
  HASH 8741753e488a78dd0d0024c980e1fb5b5c85888447e309d9cb9d949bdb52aa3e
)
lg_fetch(
  lodepng
  URL
    https://github.com/lvandeve/lodepng/archive/ed6fe5825c6a4fbb7f58ab35a4231c7543cd452a.tar.gz
  HASH c2459a3f9145258f901d262576f7a56ca08087d3b3efeee3ae033c0952120803
)
lg_fetch(
  cccl
  URL https://github.com/NVIDIA/cccl/archive/refs/tags/v3.4.2.tar.gz
  HASH 544f5f14c59cae6cfdad3156aa720f2dd44fbd6feb3556d10bdb4fe6aa3cdf4b
)
lg_fetch(
  cutlass
  URL https://github.com/NVIDIA/cutlass/archive/refs/tags/v4.7.1.tar.gz
  HASH 8290eb914cd5aaf4c665ee4108ba5bd65383cfee1296286a42a7ef711554d365
)
lg_fetch(
  nvbench
  URL
    https://github.com/NVIDIA/nvbench/archive/410dcdd21c9b48191ecb3d3d77060b1bf4ac6244.tar.gz
  HASH a8565c504462c69913f6c8e1cece76fd013980406694d8899f37aba75b04da19
)

set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)

# Header INTERFACE only. CUTLASS still enables tools unless these are off.
set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_TOOLS OFF CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_LIBRARY OFF CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_PROFILER OFF CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_CUBLAS OFF CACHE BOOL "" FORCE)
set(CUTLASS_ENABLE_CUDNN OFF CACHE BOOL "" FORCE)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(NVBench_ENABLE_CUPTI OFF CACHE BOOL "" FORCE)
set(NVBench_ENABLE_EXECUTABLES OFF CACHE BOOL "" FORCE)
set(NVBench_ENABLE_INSTALL_RULES OFF CACHE BOOL "" FORCE)
set(NVBench_ENABLE_NVML ON CACHE BOOL "" FORCE)
set(NVBench_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(rapids-cmake-sha edc8773f660e15fa2948324df5dc1a1bb7d9d1cc)

FetchContent_MakeAvailable(
  nvbench
  spdlog
  lodepng
  cccl
  cutlass
)

find_package(CUDAToolkit REQUIRED)

add_library(lg_cuda_runtime INTERFACE)
target_link_libraries(
  lg_cuda_runtime
  INTERFACE CUDA::cudart_static ${CMAKE_DL_LIBS}
)

add_library(lg_npp INTERFACE)
target_compile_definitions(lg_npp INTERFACE LG_HAS_NPP)
target_link_libraries(lg_npp INTERFACE CUDA::nppif_static)

# Oracle GEMM. Header-only: one instantiated kernel instead of the ~280 MB of
# precompiled kernels cuBLAS links in.
add_library(lg_cutlass INTERFACE)
target_compile_definitions(lg_cutlass INTERFACE LG_HAS_CUTLASS)
target_link_libraries(lg_cutlass INTERFACE nvidia::cutlass::cutlass)
# CUTLASS's CMake puts this on its own apps, not on nvidia::cutlass::cutlass.
target_compile_options(
  lg_cutlass
  INTERFACE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>"
)

# Library baseline, not an oracle. Shared, so the GPU host's toolkit supplies
# it; static cuBLAS adds about 230 MB of host code and kernel metadata per app.
add_library(lg_cublas INTERFACE)
target_compile_definitions(lg_cublas INTERFACE LG_HAS_CUBLAS)
target_link_libraries(lg_cublas INTERFACE CUDA::cublas)

add_library(lg_nvbench INTERFACE)
target_compile_definitions(lg_nvbench INTERFACE LG_HAS_NVBENCH)
target_link_libraries(lg_nvbench INTERFACE nvbench::nvbench)

# Runs every build; a stale binary then shows an old revision.
set(LG_REVISION_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/revision.cpp")
add_custom_target(
  lg_revision_stamp
  COMMAND
    "${CMAKE_COMMAND}" "-DSOURCE_DIR=${PROJECT_SOURCE_DIR}"
    "-DOUTPUT=${LG_REVISION_SOURCE}" -P
    "${PROJECT_SOURCE_DIR}/src/cmake/build_revision.cmake"
  BYPRODUCTS "${LG_REVISION_SOURCE}"
  VERBATIM
)
add_library(lg_revision STATIC "${LG_REVISION_SOURCE}")
target_include_directories(lg_revision PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
add_dependencies(lg_revision lg_revision_stamp)

add_library(
  lg_benchmark
  STATIC
  benchmark/format.cpp
  benchmark/profiler.cpp
  benchmark/runner.cu
)
target_include_directories(lg_benchmark PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(lg_benchmark PRIVATE LG_HAS_CUDA_DRIVER LG_HAS_CUPTI)
target_include_directories(
  lg_benchmark
  SYSTEM
  PRIVATE "${CPM_PACKAGE_nlohmann_json_SOURCE_DIR}/include"
)
# Shared CUPTI comes from the GPU host's toolkit; static adds about 35 MB per
# app.
target_link_libraries(
  lg_benchmark
  PUBLIC lg_nvbench lg_runtime CCCL::CCCL fmt::fmt
  PRIVATE lg_revision CUDA::cuda_driver CUDA::cupti
)

# Driver API reads kernels that libraries launch inside a captured graph.
add_library(lg_benchmark_captured STATIC benchmark/captured.cu)
target_compile_definitions(lg_benchmark_captured PRIVATE LG_HAS_CUDA_DRIVER)
target_link_libraries(
  lg_benchmark_captured
  PUBLIC lg_benchmark CCCL::CCCL
  PRIVATE CUDA::cuda_driver
)

add_library(lg_runtime STATIC log.cpp)
target_include_directories(lg_runtime PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(lg_runtime PUBLIC spdlog::spdlog)

add_library(lodepng STATIC "${lodepng_SOURCE_DIR}/lodepng.cpp")
target_include_directories(lodepng PUBLIC "${lodepng_SOURCE_DIR}")

add_library(lg_image STATIC image.cpp)
target_include_directories(lg_image PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(
  lg_image
  PUBLIC lg_runtime CUDA::cudart_static CCCL::CCCL
  PRIVATE lodepng
)

add_library(
  lg_convolution
  STATIC
  convolution/app.cpp
  convolution/correctness.cu
)
target_include_directories(lg_convolution PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(
  lg_convolution
  PUBLIC lg_runtime lg_cuda_runtime CCCL::CCCL
  PRIVATE lg_npp
)

add_library(lg_convolution_nvbench STATIC convolution/benchmark.cu)
target_link_libraries(lg_convolution_nvbench PUBLIC lg_convolution lg_benchmark)

add_library(lg_convolution3d STATIC convolution3d/correctness.cu)
target_include_directories(
  lg_convolution3d
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(
  lg_convolution3d
  PUBLIC lg_runtime lg_cuda_runtime CCCL::CCCL
)

add_library(lg_convolution3d_nvbench STATIC convolution3d/benchmark.cu)
target_link_libraries(
  lg_convolution3d_nvbench
  PUBLIC lg_convolution3d lg_benchmark
)

add_library(lg_stencil STATIC stencil/correctness.cu)
target_include_directories(lg_stencil PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(lg_stencil PUBLIC lg_runtime lg_cuda_runtime CCCL::CCCL)

add_library(lg_stencil_nvbench STATIC stencil/benchmark.cu)
target_link_libraries(lg_stencil_nvbench PUBLIC lg_stencil lg_benchmark)

add_library(
  lg_matmul
  STATIC
  matmul/app.cpp
  matmul/correctness.cu
  matmul/matrix.cpp
)
target_include_directories(lg_matmul PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(
  lg_matmul
  PUBLIC lg_runtime lg_cuda_runtime CCCL::CCCL
  PRIVATE lg_cutlass
)

add_library(lg_matmul_nvbench STATIC matmul/benchmark.cu)
target_link_libraries(lg_matmul_nvbench PUBLIC lg_matmul lg_benchmark)

if(LG_BUILD_TESTS)
  add_subdirectory(tests)
endif()
