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
  $O(R^2)$, $\lbrace \alpha, \beta\rbrace$ vs `matmul.cu`, `device::Gemm`,
  `dim3(16, 16)`.
- GitHub runs KaTeX after the markdown pass, so keep math out of the markdown
  parser's way ([writing math][gh-math]):
  - Display math goes in a ` ```math ` fence, never `$$`. A fence is the only
    place where `\\`, `\,`, and other backslash-punctuation survives.
  - Inline `$...$` stays on one line with no padding inside the delimiters, and
    spells braces `\lbrace \rbrace`. `$ x $` and `$\{x\}$` render as literals.
  - Inline, only `\` plus letters and `\` (backslash-space) reach KaTeX intact;
    backslash-punctuation loses its backslash. Space a unit with
    `$100\ \mathrm{ns}$`; `$100\,\mathrm{ns}$` renders as `100,ns`.
  - Keep `%` out of math entirely. `\%` arrives as a bare `%`, which opens a
    KaTeX comment and swallows the rest of the expression. Write `$90$%`.
  - No `\hline`, so no ruled `array`. A row-ending `\\` comes back as `\\\`, and
    that stray backslash pushes `\hline` off the start of its row, so KaTeX
    fails with `Misplaced \hline`. Build a boxed row from `\boxed` cells.
    Mid-line `\\[1em]` is left alone.
  - Verify by resolving the markdown escapes yourself, then rendering with the
    `katex` npm package. GitHub's `/markdown` API returns the exact string it
    hands to KaTeX.
  - Lay multi-matrix figures out with one `\begin{array}`, not an HTML table:
    GitHub hoists math out of `<td>` and drops the rows (see Ex 3.1).
  - Mark cells with `\boxed`. GitHub's KaTeX build drops `\colorbox`, and a
    fixed background color would fight the dark theme anyway.
- Later perf improvements rewrite reminders live in `docs/pmpp/todo.md` under a
  plain heading (`## Matmul`) so the chapter checklist can link
  `todo.md#matmul`.
- Exercise files (`docs/pmpp/chXX-ex.md`): header each as `## Ex 3.1`
  (chapter.exercise) for cross-file links. Leave an `**Answer:**` placeholder.
  Do not fill answers. Do not delete the user's answers.
- Prefer a list when the content is mostly prose. Use a table only for
  numerical comparisons.

[gh-math]: https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions

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
