cmake_minimum_required(VERSION 3.25)
project(cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
# Include native code and forward-compatible PTX.
set(CMAKE_CUDA_ARCHITECTURES all)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# Generate compile commands for host tooling.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

add_executable(main src/main.cu)

option(
  STRIP_DEPLOY
  "Strip symbols from the binary for smaller, faster deploys"
  ON
)
if(STRIP_DEPLOY AND CMAKE_STRIP)
  add_custom_command(
    TARGET main
    POST_BUILD
    COMMAND "${CMAKE_STRIP}" "$<TARGET_FILE:main>"
    COMMENT "Stripping binary for deploy (disable with -DSTRIP_DEPLOY=OFF)"
    VERBATIM
  )
endif()
