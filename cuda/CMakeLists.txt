cmake_minimum_required(VERSION 3.25)
project(cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
# Include native code and forward-compatible PTX.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# Generate compile commands for host tooling.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

function(add_kernel target architectures source)
  add_library(${target} OBJECT ${source})
  set_target_properties(
    ${target}
    PROPERTIES CUDA_ARCHITECTURES "${architectures}"
  )
  target_include_directories(${target} PRIVATE src/common)
endfunction()

add_kernel(kernel_hello "86;90;100" src/common/hello.cu)
add_kernel(kernel_ampere 86 src/sm86/async_copy.cu)
add_kernel(kernel_hopper 90a src/sm90a/cluster.cu)
add_kernel(kernel_blackwell 100a src/sm100a/fp4.cu)

add_executable(
  main
  src/common/main.cu
  src/common/registry.cu
  $<TARGET_OBJECTS:kernel_hello>
  $<TARGET_OBJECTS:kernel_ampere>
  $<TARGET_OBJECTS:kernel_hopper>
  $<TARGET_OBJECTS:kernel_blackwell>
)
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "86;90;100")
target_include_directories(main PRIVATE src/common)

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
