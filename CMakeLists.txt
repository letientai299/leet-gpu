cmake_minimum_required(VERSION 3.30.4)
project(cuda LANGUAGES CXX CUDA)

include(FetchContent)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  # Empty type is -O0 and keeps host debug sections; Release + strip is the
  # deploy path. CLion still sets Debug in its own build tree.
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# nvcc compiles each CUDA_ARCHITECTURES entry in one invocation; -t0 uses
# every core for those device passes.
if(
  CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA"
  AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.6
)
  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-t0>")
endif()

# Drop unused host objects pulled in by static libstdc++ / spdlog.
# https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#index-ffunction-sections
if(
  CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
  OR CMAKE_CXX_COMPILER_ID MATCHES "Clang"
)
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:-ffunction-sections>"
    "$<$<COMPILE_LANGUAGE:CXX>:-fdata-sections>"
    "$<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffunction-sections,-fdata-sections,-fvisibility=hidden>"
  )
endif()

# Default fatbin compression favors speed. size costs extra load-time inflate.
# Needs a driver from CUDA 12.4+.
# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#compress-mode
if(
  CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA"
  AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8
)
  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--compress-mode=size>")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# `86` also embeds PTX (`86-virtual`). Cubin-only keeps the three SASS
# images the deploy hosts actually run.
set(
  LEET_GPU_CUDA_ARCHITECTURES
  "86-real;90-real;100-real"
  CACHE STRING
  "Default CUDA architectures"
)
set(
  CMAKE_CUDA_ARCHITECTURES
  "${LEET_GPU_CUDA_ARCHITECTURES}"
  CACHE STRING
  "CUDA architectures"
  FORCE
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
