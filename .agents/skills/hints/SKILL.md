---
name: hints
description: >-
  Monitor example code and exercise files for saves, then hint one issue on a
  complete unit of work. Use when the user invokes $hints or /hints, asks for
  hints or save monitoring, or works on PMPP exercises or kernels as a study
  session.
metadata:
  compatibility: Codex, Cursor, and Claude Code; requires watchexec.
---

# Hints

Monitor file changes and inspect only work saved since the last wake. Hint only
after a changed **unit of work** looks complete. A unit is a sub-problem, e.g.
3.4a, 3.4b, or a block of code to complete a particular logic, not necessarily
a whole exercise, function or file. Show feedback soon after the save so it is
glanceable in the agent terminal.

This is a study project. The user does the exercises. Leave `**Answer:**`
placeholders empty. Do not delete filled answers. Do not edit their files.

Local IDE/CLI only. Cloud timers cannot watch the filesystem.

## Parse

- Codex `$hints`; Cursor or Claude Code `/hints` — watch default roots
  `docs/pmpp` and `src/pmpp`.
- `$hints <path>…` or `/hints <path>…` — watch those paths instead.
  Named directory: `-w` that directory. Named file: `-w` its parent and
  `--filter` the exact repo-relative path. Never `-w` the file itself.
- `$hints stop` or `/hints stop` — stop the tracked monitor; do not re-arm.
- A plain-language request to monitor exercises or provide hints follows the
  same flow without requiring explicit invocation syntax.

## The watcher command

One command for every client. It prints **one** `hints-save:` line per
debounced batch. A noisy watcher (one stdout line per raw event) is the usual
reason wakes stop arriving. An atomic save is `modify:` plus `rename:` for the
same path in one batch; the sentinel keeps that as a single wake.

`--postpone` skips a startup run. `--no-meta` ignores chmod/atime.
`--emit-events-to=stdio` pipes `kind:path` events to the command's stdin.
`sort -u` collapses the batch. Keep `--debounce=1s`: that is the wake target and
also the floor that holds an atomic save's `modify` + `rename` in one batch.
Below `1s` the two halves split into two wakes and the first reads a
partially written file.

`--exts` and `--filter` are **OR**. Use `--exts` only on the default roots,
with no `--filter`. Whenever any `--filter` is set, drop `--exts` or every
sibling `.cu` / `.md` in the watched parent also fires.

Repo root. Default roots:

```bash
mise exec -- watchexec \
  --postpone --quiet --no-meta --debounce=1s \
  --exts=md,cu,cuh,hpp,h,cpp,cc,c \
  --emit-events-to=stdio --shell=sh \
  -w docs/pmpp -w src/pmpp \
  -- 'echo "hints-save: $(sort -u | tr "\n" " ")"'
```

Named file. Watch the parent, filter the file, no `--exts`:

```bash
mise exec -- watchexec \
  --postpone --quiet --no-meta --debounce=1s \
  --emit-events-to=stdio --shell=sh \
  -w src/pmpp/ch03 --filter 'src/pmpp/ch03/matmul.row.cu' \
  -- 'echo "hints-save: $(sort -u | tr "\n" " ")"'
```

Filter patterns are repo-relative. Multiple named paths: one `-w` per directory
or file parent, and one `--filter` per named path (exact file path, or
`<dir>/**` for a directory). If any path needs a filter, every named path gets
one, and `--exts` is omitted.

Output is one line per batch:

```text
hints-save: modify:/abs/path/matmul.row.cu remove:/abs/path/ch03-ex.md
```

## Select the monitor

Inspect the available tool schemas. Use the first supported event-driven route,
running the matching command from above:

1. A persistent `Monitor` tool: pass the command, a short description, and
   `persistent: true`. Each `hints-save:` line is one wake. Claude Code may
   expose this route.
2. A shell with output notifications: start in the background with
   `block_until_ms: 0` and register `notify_on_output` with:

   ```text
   pattern: ^hints-save:
   reason: hints file change
   debounce_ms: 5000
   ```

   Anchor the pattern and set `debounce_ms`. A pattern that matches every event
   line gets the watcher dropped as noisy. Cursor may expose this route.
   `debounce_ms` is the minimum gap *between* notifications, not a delay added
   to each one: an isolated save after an idle stretch wakes as soon as the
   sentinel line lands. It only bites on back-to-back saves. `5000` is the
   harness minimum; lower values are silently raised to it.
3. A shell without output notifications: append `; kill -TERM "$PPID"` inside
   the command string and run it as a foreground long-running call. It forwards
   one batch, terminates itself, and returns the batch as the tool result. Use
   the longest supported timeout and re-arm after a timeout or event. If the
   client moves long calls into the background, retain its task handle and react
   to the completion notification. Every CLI/IDE agent can run this route.

Do not assume a parameter exists because another client supports it. Do not use
`/loop` as the primary wake, busy-poll, or start a detached process whose output
cannot reach this session. If none of these routes exists, report that immediate
save watching is unavailable instead of claiming the watch is active.

## Arm (once)

1. Check monitors owned by this session for `watchexec` with the same repo,
   route, and watch roots. Reuse an exact match. Do not reuse another session's
   monitor or a watcher with different roots.
2. Start the command through the selected route. When supported, title the task
   `Hints watch: pmpp`.
3. Smoke-check once that the process is running. It stays quiet until a
   save. Do not review the tree on startup.
4. Confirm in one short line: watching, which roots, that saves wake this
   session, and that **no message after a save means the unit still looks like
   a draft**.
5. Initialize session state: an empty `seen` snapshot, an empty `pending` queue,
   and the monitor task handle or PID.

On later wakes, re-arm an exited watcher through the same route. If both output
and completion notifications arrive, act on the output once. Do not write a
custom file-watch script.

## On each wake

A wake is a notification, a returned batch, **or** any user message while the
watch is armed — output notifications only land after a turn ends, so a batch
can arrive late or coalesce with the next one. Always read the watcher output
from where you last read it, not only the newest line.

Split each unread `hints-save:` line on spaces into `kind:path` records. Make
paths repo-relative and drop directories.

- `remove`: invalidate its `seen` snapshot and any pending units. Do not read
  the missing path. A pure deletion produces no hint.
- `rename`: stat all reported paths. Treat existing paths as destinations to
  inspect and missing paths as removals to invalidate.
- `create`, `modify`, or `other`: inspect only paths that exist.

The remaining existing paths are `changed`.

If the watcher recorded batches you never got notified about, treat them as
unread now. If the user asks why a save was silent, check the watcher is alive
and say whether the batch was logged, instead of assuming the unit was a draft.

- `changed` empty → no-op. Do not review. Do not write.
- Else: read **only** those paths (plus kernels newly linked from a changed
  exercise). Do not re-read unchanged files.

Spend **one** tool call on the read, batching the paths in a single message. The
batch line already carries the kind and path, so a separate `stat` for mtimes
buys nothing and each extra round trip costs more wall clock than both
debounces combined. Grep for kernels linked from an exercise only the first
time that exercise is seen.

Inspect the diff versus what you last read. If a path is new to you, the whole
current contents of the changed unit is in scope — not the entire file.

Then classify each **unit of work** that actually changed. Identity is the
sub-problem, not the enclosing `## Ex`, function, or file. Refresh or remove
pending units that overlap a changed path **and** the same sub-problem.

- Exercise: one lettered part under `## Ex X.Y` (`- a.` → Ex 3.4a). No
  lettered parts → the whole `## Ex X.Y` answer. Sibling parts are separate
  units.
- Code: one finished logic block in `src/pmpp` (thread/row/col index, bounds
  check, inner product, store, launch config, host stub). Split on comments,
  loops, guards, or launch. Not the whole function or file.

A later empty sibling (Ex 3.4b still blank, GEMM loop still `TODO`) does not
keep an already-finished unit as a draft.

### Draft → stay silent

Stay silent. Do not nudge to finish. Do not say "still drafting."

Treat **that unit** as draft if any of:

- Unclosed fence, `$…$`, or braces in the unit (mid-save).
- Exercise part: `**Answer:**` missing that part, or the part is whitespace /
  `TODO` / `…` / `TBD`.
- Code block empty, comments-only, or only `TODO` / `FIXME`.
- Answer is only a link to a kernel whose matching logic block is still a
  draft.

Do not wait for other parts of the same exercise or the rest of the function.

### Complete → one glanceable hint

Review only complete units. Write **once**, short enough to read in the
agent terminal after a save (about 6 lines). Do not paste a full solution
or a rewritten kernel.

```text
hints · <Ex X.Ya or function · logic>

<one issue, or one improvement if correct>
```

- Wrong or incomplete reasoning: **one** issue. Point at the mismatch
  (index, launch config, bounds, algorithm). Do not give the full solution.
- Correct: optionally **one** further improvement — perf, alternative idea,
  or a production-grade CUDA lib / framework. Skip if you already gave that
  improvement for this unit and the code did not change.

Add every complete changed unit to `pending`, newest content replacing older
content for the same unit. On each wake, hint one pending unit: prefer a wrong
unit, then the unit that changed most. Remove it from `pending` after hinting.
Unhinted units remain eligible on the next wake even when unchanged.

If no unit is pending or a complete unit is unchanged since its last hint, stay
silent.

## Stop

Stop only the tracked task handle or PID with the selected client's task-stop
mechanism. Await or drain its completion so it cannot wake the session again.
Confirm stopped. Do not re-arm.
