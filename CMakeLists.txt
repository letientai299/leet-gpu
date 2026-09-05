cmake_minimum_required(VERSION 3.22)
project(cuda LANGUAGES CXX CUDA)

include(FetchContent)

# nvcc compiles each CUDA_ARCHITECTURES entry in one invocation; -t0 uses
# every core for those device passes.
if(
  CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA"
  AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.6
)
  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-t0>")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(
  LEET_GPU_CUDA_ARCHITECTURES
  "86;90;100"
  CACHE STRING
  "Default CUDA architectures"
)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

option(STRIP_DEPLOY "Strip symbols from deploy binaries" ON)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/src/cmake")
include(AddCudaApp)

add_subdirectory(src/libs)
add_subdirectory(src/pmpp)
