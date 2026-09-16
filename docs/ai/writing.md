# Writing

Keep comments, commit bodies, and docs short, concise, direct, and imperative.

## Commits

- Use conventional commits with no scope.
- Wrap the commit message body at 72 characters.

## Comments

- Explain why, not what the code already says.
- Link to external guides, APIs, or documentation for reference.

## Docs

- When adding a kernel under `src/`, link it from the matching chapter file in
  `docs/pmpp/chNN.md` (see 2.3 / 3.2). Keep `docs/pmpp/readme.md` as an index.
  Do not catalog apps in the root `readme.md`. Keep setup, workflow, and editor
  notes in `docs/readme.md`.
- Use reference-style links in Markdown files to prevent broken prose lines.
  Keep link definitions below and near the referencing paragraphs. Use inline
  links for the short relative routes in `docs/pmpp/readme.md`.
- Write math with `$...$` and use display math when it helps. Backticks are for
  code: paths, APIs, types, and launch configuration. Example:
  $C$, $A[\mathrm{row}, i]$, $O(R^2)$, $\lbrace \alpha, \beta\rbrace$ versus
  `matmul.cu`, `device::Gemm`, and `dim3(16, 16)`.
- GitHub runs KaTeX after the Markdown pass, so keep math out of the Markdown
  parser's way ([writing math][gh-math]):
  - Put display math in a ` ```math ` fence, never `$$`. A fence is the only
    place where `\\`, `\,`, and other backslash-punctuation survives.
  - Keep inline `$...$` on one line with no padding inside the delimiters, and
    spell braces `\lbrace \rbrace`. `$ x $` and `$\{x\}$` render as literals.
  - Inline, only `\` plus letters and `\` followed by a space reach KaTeX
    intact. Backslash-punctuation loses its backslash. Space a unit with
    `$100\ \mathrm{ns}$`; `$100\,\mathrm{ns}$` renders as `100,ns`.
  - Keep `%` out of math. `\%` arrives as a bare `%`, which opens a KaTeX
    comment and swallows the rest of the expression. Write `$90$%`.
  - Do not use `\hline`, so do not use a ruled `array`. A row-ending `\\`
    comes back as `\\\`, and that stray backslash pushes `\hline` off the
    start of its row. KaTeX then fails with `Misplaced \hline`. Build a boxed
    row from `\boxed` cells. Mid-line `\\[1em]` is left alone.
  - Verify by resolving the Markdown escapes, then rendering with the `katex`
    npm package. GitHub's `/markdown` API returns the exact string it passes to
    KaTeX.
  - Lay out multi-matrix figures with one `\begin{array}`, not an HTML table.
    GitHub hoists math out of `<td>` and drops the rows (see Ex 3.1).
  - Mark cells with `\boxed`. GitHub's KaTeX build drops `\colorbox`, and a
    fixed background color conflicts with the dark theme.
- Keep later performance rewrite reminders in `docs/pmpp/todo.md` under a plain
  heading such as `## Matmul`, so the chapter checklist can link
  `todo.md#matmul`.
- Header exercise sections in `docs/pmpp/chNN.md` as `## Ex 3.1`
  (chapter.exercise) for cross-file links. Leave an `**Answer:**` placeholder.
  Do not fill answers. Do not delete the user's answers.
- Prefer a list when the content is mostly prose. Use a table only for numerical
  comparisons.

[gh-math]:
  <https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions>
