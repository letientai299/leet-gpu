# Project Agent Rules

## About

This is a study project. Most local code and notes reflect the user's current
understanding and might be incorrect. Do not read them unless the user asks
about them. Base answers on books, official CUDA guides, and verified public
knowledge instead.

When adding C++ or CUDA libraries, update `.clangd` for required include paths
or flags missing from the compilation database. Check an affected translation
unit with container clangd and run `mise run lsp:check`.

## Exercises

On exercises and learning examples, wait for the user. Reply in chat. Do not
write a review file unless asked. Do not fill `**Answer:**` placeholders or
write solutions into the user's files. Do not delete filled answers. Do not edit
the user's files.

- **Hint** (`hint`): Say **Correct** or **Wrong** immediately. If wrong, hint
  every issue. Do not give the solution.
- **Check** (`review`, `check`, `validate`): Say **Correct** or **Wrong**
  immediately, then give the solution.
- For multiple labeled subquestions, mark every subquestion **Correct** or
  **Wrong**. If an answer contains any error, mark it **Wrong**, then identify
  its correct and incorrect parts.

### Other rules

Read every file whose triggers match the task. Do not read unmatched files.

- **Docs, Markdown, math, KaTeX, code, comment, commit message:** Read
  [`docs/ai/writing.md`](docs/ai/writing.md).
- **Oracle, reference implementation, CUTLASS, cuBLAS, GEMM correctness:** Read
  [`docs/ai/oracles.md`](docs/ai/oracles.md).
