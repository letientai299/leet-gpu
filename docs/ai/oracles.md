# Oracles

Check a handwritten kernel against an NVIDIA library. This also shows how to
call the library. See `src/pmpp/ch03/matmul.cu`.

- Do not call the oracle from the kernel under test.
- Prefer a header-only library such as CUTLASS over a precompiled toolkit
  library. `deploy` copies only the binary, so a static toolkit library lands in
  every app; static cuBLAS adds about 230 MB. A speed baseline such as
  `matmul.cublas` links the GPU host's shared cuBLAS instead (see `AGENTS.md`).
- Wrap each library with an INTERFACE target in `src/libs`.
  `find_package(CUDAToolkit)` lives there because imported `CUDA::*` targets are
  not visible to `src/pmpp`. Link only apps that need the library.
- Put status checks and RAII handles in `checks.hpp`. Gate a library's headers
  with a compile definition from its INTERFACE target so other apps skip them.
  Prefer CCCL for allocation and copying (`cuda::device_buffer`,
  `cuda::copy_bytes`).
- For GEMM, instantiate CUTLASS 2.x `cutlass::gemm::device::Gemm`, not 3.x CuTe
  or `GemmUniversalAdapter`. Use row-major to match the kernel (`lda` is $k$ for
  $A$, width for $B$ and $C$). Default `OpClassSimt` is CUDA-core FFMA (IEEE
  FP32), not Tensor Core TF32. Keep learner API links in `docs/pmpp/ch03.md`
  section 3.4, not as a URL dump in the `.cu` file.
- Match the naive kernel's precision and layout. Vendor defaults may use reduced
  precision or a different memory order. For floats, compare with `atol` and
  `rtol`, not bitwise `==`.
- Use the host STL to fill and verify small inputs. Do not add CUDA libraries
  for that. Stay on C++20; `nvcc` has no C++26 support.
