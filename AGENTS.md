# Project Agent Rules

- Use conventional commits with no scope
- Keep commit message body, comments and docs short, concise, direct,
  imperative. Wrap commit message body at 72 chars.
- When adding a kernel under `src/`, link it from the matching section in
  `docs/pmpp/readme.md` (see 2.3 / 3.2). Do not catalog apps in the root
  `readme.md`. Keep setup, workflow, and editor notes in `docs/readme.md`.
- Use ref style links in markdown files to prevent broken prose lines. Keep link
  definitions below and near the referencing paragraphs.
- When adding C++ or CUDA libraries, update `.clangd` for required include paths
  or flags missing from the compilation database. Check an affected translation
  unit with container clangd and run `mise run lsp:check`.

## Oracles

Check a handwritten kernel against a CUDA toolkit library. That also shows
how to call the library. Example: `src/pmpp/ch03/matmul.cu`.

- Do not call the oracle from the kernel under test.
- Wrap each toolkit lib with an INTERFACE target in `src/libs`.
  `find_package(CUDAToolkit)` lives there; imported `CUDA::*` targets are
  not visible to `src/pmpp` otherwise. Link only apps that need the lib.
- Put status checks and RAII handles in `checks.hpp`. Prefer CCCL for
  alloc and copy (`cuda::device_buffer`, `cuda::copy_bytes`).
- Match the naive kernel's precision and layout. Vendor defaults may use
  reduced precision or a different memory order. For floats, compare with
  atol/rtol, not bitwise `==`.
- Use host STL to fill and verify small inputs. Do not add extra CUDA
  libs for that. Stay on C++17; nvcc has no C++26.
