# Project Agent Rules

## Writing

Keep comments, commit bodies, and docs short, concise, direct, imperative.

### Commits

- Conventional commits with no scope
- Wrap the commit message body at 72 chars

### Comments

- Explain why, not what the code already says
- Add external links to guide, API, docs for referrence.

### Docs

- When adding a kernel under `src/`, link it from the matching section in
  `docs/pmpp/readme.md` (see 2.3 / 3.2). Do not catalog apps in the root
  `readme.md`. Keep setup, workflow, and editor notes in `docs/readme.md`.
- Use ref style links in markdown files to prevent broken prose lines. Keep link
  definitions below and near the referencing paragraphs.
- Write math with `$...$` (display math when it helps). Backticks are for code:
  paths, APIs, types, launch config. Example: $C$, $A[\mathrm{row}, i]$,
  $O(R^2)$, $\{\alpha, \beta\}$ vs `matmul.cu`, `device::Gemm`, `dim3(16, 16)`.
- Later perf improvements rewrite reminders live in `docs/pmpp/todo.md` under a
  plain heading (`## Matmul`) so the chapter checklist can link
  `todo.md#matmul`.
- Exercise files (`docs/pmpp/chXX-ex.md`): header each as `## Ex 3.1`
  (chapter.exercise) for cross-file links. Leave an `**Answer:**` placeholder.
  Do not fill answers. Do not delete the user's answers.
- Prefer a list when the content is mostly prose. Use a table only for
  numerical comparisons.

## Tooling

- When adding C++ or CUDA libraries, update `.clangd` for required include paths
  or flags missing from the compilation database. Check an affected translation
  unit with container clangd and run `mise run lsp:check`.

## Oracles

Check a handwritten kernel against an NVIDIA library. That also shows how to
call the library. Example: `src/pmpp/ch03/matmul.cu`.

- Do not call the oracle from the kernel under test.
- Prefer a header-only library (CUTLASS) over a precompiled toolkit lib.
  `deploy` copies only the binary, and the GPU host has no matching toolkit, so
  a toolkit lib must be static: cuBLAS alone costs ~280 MB per app.
- Wrap each lib with an INTERFACE target in `src/libs`.
  `find_package(CUDAToolkit)` lives there; imported `CUDA::*` targets are not
  visible to `src/pmpp` otherwise. Link only apps that need the lib.
- Put status checks and RAII handles in `checks.hpp`. Gate a lib's headers with
  a compile definition from its INTERFACE target so other apps skip them. Prefer
  CCCL for alloc and copy (`cuda::device_buffer`, `cuda::copy_bytes`).
- For GEMM, instantiate CUTLASS 2.x `cutlass::gemm::device::Gemm`, not 3.x CuTe
  / `GemmUniversalAdapter`. Use row-major to match the kernel (`lda` is $k$ for
  $A$, width for $B$ and $C$). Default `OpClassSimt` is CUDA-core FFMA (IEEE
  FP32), not Tensor Core TF32. Keep learner API links in `docs/pmpp/readme.md`
  §3.4, not as a URL dump in the `.cu`.
- Match the naive kernel's precision and layout. Vendor defaults may use reduced
  precision or a different memory order. For floats, compare with atol/rtol, not
  bitwise `==`.
- Use host STL to fill and verify small inputs. Do not add extra CUDA libs for
  that. Stay on C++17; nvcc has no C++26.

## Hints

Study project: the user does the exercises. Do not fill `**Answer:**`
placeholders or write exercise solutions. Do not delete filled answers. Hint
one issue per sub-problem (Ex 3.4a, one logic block); do not dump the
solution. Do not wait for a whole exercise, function, or file.

For save-watch feedback, use `$hints` in Codex or `/hints` in Cursor and Claude
Code. A plain-language request for hints also invokes the skill.
