cmake_minimum_required(VERSION 3.25)
project(cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
# Target NVIDIA A100 (compute capability 8.0 -> sm_80).
# Adjust for other GPUs: 86=Ampere consumer (RTX 30xx), 89=Ada (RTX 40xx),
# 90=Hopper (H100), 120=Blackwell. Use "80-real" for SASS only, "80-virtual"
# to embed PTX for forward-compat JIT, or a list ("80;89;90") for a fatbin.
# Single vs virtual arch: https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
# GPU feature list / arch flags: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list
# Fatbinaries & PTX/JIT: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list
#
# "80-real" emits SASS only (no embedded PTX): one less nvcc codegen pass, which
# matters because nvcc runs x86 under QEMU emulation here. Safe as long as we
# only run on real sm_80 A100s; add "80-virtual" back for forward-compat JIT.
set(CMAKE_CUDA_ARCHITECTURES 80-real)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# Emit build/compile_commands.json so an IDE/clangd on the host can index the
# code against the real CUDA headers, even though nvcc only exists in the
# container. Paths inside are container paths (/app/...); map them in the IDE.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Cache host-side compilation so reconfigures, reverts, and branch switches
# don't re-run the emulated compiler. No-op when ccache isn't installed.
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

# No CUDA_SEPARABLE_COMPILATION: a single translation unit doesn't need
# relocatable device code, and enabling it adds an extra nvcc device-link pass.
add_executable(main src/main.cu)

# Strip host symbols/debug info before deploy: they dominate binary growth as
# the code grows and aren't needed to run on the GPU (device code lives in
# dedicated ELF sections that strip leaves intact). Runs only on relink, so it
# doesn't touch the binary on no-op builds. Turn off with -DSTRIP_DEPLOY=OFF
# when you need symbols for cuda-gdb on the a100.
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
