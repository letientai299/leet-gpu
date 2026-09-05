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
