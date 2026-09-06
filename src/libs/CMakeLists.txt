set(LEET_GPU_DEP_CACHE "${CMAKE_SOURCE_DIR}/.cache/fetchcontent")

set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)

# Keep extracted sources outside build/ so wiping the CMake tree skips
# re-download. Binary dirs stay per-prefix (build/docker vs build/local).
FetchContent_Declare(
  spdlog
  URL https://github.com/gabime/spdlog/archive/refs/tags/v1.16.0.tar.gz
  URL_HASH
    SHA256=8741753e488a78dd0d0024c980e1fb5b5c85888447e309d9cb9d949bdb52aa3e
  SOURCE_DIR "${LEET_GPU_DEP_CACHE}/spdlog-src"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_Declare(
  lodepng
  URL
    https://github.com/lvandeve/lodepng/archive/ed6fe5825c6a4fbb7f58ab35a4231c7543cd452a.tar.gz
  URL_HASH
    SHA256=c2459a3f9145258f901d262576f7a56ca08087d3b3efeee3ae033c0952120803
  SOURCE_DIR "${LEET_GPU_DEP_CACHE}/lodepng-src"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_Declare(
  cccl
  URL https://github.com/NVIDIA/cccl/archive/refs/tags/v3.4.2.tar.gz
  URL_HASH
    SHA256=544f5f14c59cae6cfdad3156aa720f2dd44fbd6feb3556d10bdb4fe6aa3cdf4b
  SOURCE_DIR "${LEET_GPU_DEP_CACHE}/cccl-src"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_Declare(
  cutlass
  URL https://github.com/NVIDIA/cutlass/archive/refs/tags/v4.7.1.tar.gz
  URL_HASH
    SHA256=8290eb914cd5aaf4c665ee4108ba5bd65383cfee1296286a42a7ef711554d365
  SOURCE_DIR "${LEET_GPU_DEP_CACHE}/cutlass-src"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_Declare(
  nvbench
  URL
    https://github.com/NVIDIA/nvbench/archive/410dcdd21c9b48191ecb3d3d77060b1bf4ac6244.tar.gz
  URL_HASH
    SHA256=a8565c504462c69913f6c8e1cece76fd013980406694d8899f37aba75b04da19
  SOURCE_DIR "${LEET_GPU_DEP_CACHE}/nvbench-src"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

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
set(NVBench_ENABLE_NVML OFF CACHE BOOL "" FORCE)
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

# Oracle GEMM. Header-only: one instantiated kernel instead of the ~280 MB of
# precompiled kernels cuBLAS links in.
add_library(leet_gpu_cutlass INTERFACE)
target_compile_definitions(leet_gpu_cutlass INTERFACE LEET_GPU_HAS_CUTLASS)
target_link_libraries(leet_gpu_cutlass INTERFACE nvidia::cutlass::cutlass)
# CUTLASS's CMake puts this on its own apps, not on nvidia::cutlass::cutlass.
target_compile_options(
  leet_gpu_cutlass
  INTERFACE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>"
)

add_library(leet_gpu_nvbench INTERFACE)
target_compile_definitions(leet_gpu_nvbench INTERFACE LEET_GPU_HAS_NVBENCH)
target_link_libraries(leet_gpu_nvbench INTERFACE nvbench::nvbench)

add_library(leet_gpu_runtime STATIC log.cpp)
target_include_directories(
  leet_gpu_runtime
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(leet_gpu_runtime PUBLIC spdlog::spdlog)

add_library(lodepng STATIC "${lodepng_SOURCE_DIR}/lodepng.cpp")
target_include_directories(lodepng PUBLIC "${lodepng_SOURCE_DIR}")

add_library(leet_gpu_image STATIC image.cpp)
target_include_directories(leet_gpu_image PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(
  leet_gpu_image
  PUBLIC leet_gpu_runtime CUDA::cudart_static CCCL::CCCL
  PRIVATE lodepng
)
