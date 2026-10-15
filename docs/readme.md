# Setup

## Prerequisites

[mise][mise] for tools and tasks.

[mise]: https://mise.jdx.dev

### macOS

No local `nvcc`. Docker builds the `linux/amd64` image (`mise run dc:up`). Set
`SSH_HOST` to a GPU machine to run binaries. Homebrew `rsync` 3.2+ is preferred
over macOS `openrsync` for `deploy`.

### Linux GPU host

NVIDIA driver, CUDA toolkit (`nvcc`), CMake 3.22+, Ninja 1.12+, and ccache. With
`BUILD_ENV=auto` and empty `SSH_HOST`, build and run locally. `build` looks for
`nvcc` on `PATH`, then under `CUDACXX`, `CUDA_HOME`, `CUDA_PATH`, and a toolkit
prefix such as `/usr/local/cuda`, and exports `CUDACXX` for CMake; point
`CUDACXX` at any other install. Set
[`LG_CUDA_ARCHITECTURES`][cmake-lists] to cubin-only targets for the GPUs you
run on; a bare number also embeds PTX. `mise build` configures Release,
compresses fatbins, GC-sections unused host objects, and strips symbols.
Binaries from Docker use CUDA 13.3 and a static runtime; the host driver must
load them (CUDA 12.4+ for `--compress-mode=size`).

[cmake-lists]: ../CMakeLists.txt

`deploy` copies only the binary, so an app must not need a CUDA shared library
at run time that the host toolkit lacks. Benchmark apps load the host's
`libcupti.so.13`, and cuBLAS apps its `libcublas.so.13`. The run host has its
own toolkit, and its version rarely matches the container's.

## Environment

```sh
cp .env.example .env
```

`BUILD_ENV` accepts `auto`, `local`, or `docker`. `auto` builds locally when
`nvcc` exists and uses Docker otherwise. CMake writes to `build/local` or
`build/docker`; CLion configures its own tree at `build/clion`. Binaries still
land in `bin/`. Fetched dependency sources live in `.cache/fetchcontent/`;
compiler caches live under the matching build directory. `mise clean` resets
first-party CMake state, objects, and binaries while retaining fetched and built
dependencies and compiler caches. `mise clean:all` removes all generated state.
`SSH_HOST` independently selects the run host; leave it empty for local
execution. A non-empty shell value overrides the fallback in `.env`.

## Workflow

Build every app or one app:

```sh
mise run build
mise run build -- src/pmpp/ch03/grayscale
```

Each source path under `src/` defines its app key and binary path. For example,
`src/pmpp/ch03/grayscale.cu` builds `bin/pmpp/ch03/grayscale`. The `src/` prefix
is optional. Use it in `.env` for editor path completion.

Deploy one app and forward its arguments:

```sh
mise run deploy -- src/pmpp/ch02/hello
mise run deploy -- src/pmpp/ch03/grayscale -i /data/input.png -o /data/output.png
```

Build and deploy one app:

```sh
mise run try -- src/pmpp/ch03/grayscale -i /data/input.png -o /data/output.png
```

Remote argument paths refer to files on `SSH_HOST`. `APP` supplies the default
app for `deploy`, `try`, and `dev`. `ARGS` supplies trusted shell-word arguments
when no arguments are passed explicitly.

Watch the selected app and shared code, then rebuild and deploy after changes:

```sh
mise run dev -- src/pmpp/ch03/grayscale -i /data/input.png -o /data/output.png
mise run dev
```

`dev` also watches `.env`, so editing `APP` or `ARGS` retargets the running
loop.

Every binary supports `--help` without initializing CUDA.

With `--bench`, `deploy`, `try`, and `dev` lock the target GPU's SM clock so
timings compare at one clock. The default picks the highest supported clock at
or below 85% of the maximum, under the point where the power cap throttles. A
lone `deploy` or `try` locks for its own run and unlocks afterward. `dev` keeps
the lock between runs, unlocks the old GPU when `.env` points at another target
or GPU, and unlocks on exit. The run is pinned to the locked GPU through
`CUDA_VISIBLE_DEVICES`, since `nvidia-smi` and CUDA can number GPUs differently.
Locking runs `nvidia-smi -lgc` on the target, through passwordless `sudo` or as
root; when that fails, the run continues unlocked. `GPU_CLOCK` takes `auto`, a
MHz value, or `off`; `GPU_CLOCK_FRACTION` and `GPU_INDEX` adjust `auto` and the
GPU.

An automatic lock whose sampled SM clock fell more than 1% below the lock in
over 5% of a run's samples lowers a per-GPU ceiling in
`/tmp/lg-gpu-<uuid>.clock` on the host by one supported clock; later automatic
locks stay at or below it. Delete the file to reset it. Before a locked run, an
active hardware, thermal, or power-brake slowdown stops the run. Status reads
`THROTTLED` when the measured clock falls more than 1% below the lock, and
`UNHEALTHY` (the run fails) when it stays under half the maximum clock.

Every run waits while another run holds the same GPU, through a per-GPU lock
file in the host's `/tmp`; `bench` refuses instead of waiting. A locked run
passes NVBench a `--throttle-threshold` just under the locked clock, while
unlocked runs keep NVBench's 75% check. Remote runs get a TTY, so `Ctrl-C` or a
`dev` restart stops the remote process and releases its lock.

They also pass NVBench `--stopping-criterion entropy`, which settles the median
in about a tenth of the samples that the default `stdrel` criterion takes.
`bench` keeps `stdrel`. Passing `--stopping-criterion` or any criterion
parameter such as `--max-noise` keeps NVBench's own choice.

The Status row judges stability by the cold IQR relative to the median, not the
per-sample standard deviation. Rare slow runs inflate the standard deviation
without moving the median. An IQR of one CUDA event timer step (1.024 µs on
the A100X) also counts as stable when it stays within 5% of the median, since
events cannot resolve less. Status also
prints the 95% confidence interval of the mean.

Run a benchmark under a prepared GPU, on the GPU host itself:

```sh
mise run bench -- src/pmpp/ch05/matmul.tile --bench
```

`bench` pins the process to one GPU, waits for an idle thermal state, and locks
the SM clock below the point where the power cap throttles, so repeated runs
compare at the same clock. Every change is reverted when the run ends, including
on failure or `Ctrl-C`. Locking needs root; without it the run continues and
warns. `BENCH_GPU`, `BENCH_CLOCK`, and `BENCH_IDLE_TEMP` override the defaults,
and `BENCH_CLOCK=off` skips locking entirely. The run resets the clock to the
driver default afterward; set `BENCH_RESTORE_CLOCK` to a MHz range to restore a
prior lock instead.

Comparing kernels by eye does not account for clock drift. Run the apps on one
target and print their metrics together:

```sh
mise cmp src/pmpp/ch09/hist src/pmpp/ch09/hist.private -- --axis N=1
```

`cmp` takes a baseline and one or more apps. It builds them, runs each through
`deploy` on `SSH_HOST` with `--bench --json`, then renders the results on the
target with `tools/cmp`, which takes no GPU lock. Forwarded arguments start with
`-` and go to every app; `--bench`, `--json`, `--quiet`, and listing flags are
rejected.

- Each metric prints one line per app. Rows that read the same for every app
  print once as `all`; an app without a row shows `n/a`.
- Later apps show a ratio to the baseline. Time, bytes, and rates are green when
  better and red when worse, unless the gap is under 1% or the runs' combined
  noise. Other ratios are dim.
- Apps run one after another, so a changed SM clock prints a note. The ratios
  then include that change.
- Runs pair by benchmark and state position. The command fails before printing
  when state counts or axis values differ.

For an NVBench verdict on two kernels in one saved run, use the hidden
`bench-compare` task:

```sh
mise run bench -- src/pmpp/ch05/matmul.tile --bench --jsonbin .ai/bench/run.json
mise run bench-compare -- .ai/bench/run.json
```

`bench-compare` renames both kernels to a common name so
[`nvbench-compare-robust`][nvbench-compare] will pair them — it matches
benchmarks by name and silently skips anything unpaired — and reports
`SAME`/`FAST`/`SLOW`/`AMBG`. A `FAST` verdict is only issued when the gap also
holds in cycle space, which is what makes it survive a throttled run. The tool
is a Python script shipped in the NVBench sources, not a build target; `uv`
supplies its dependencies per invocation and installs nothing.

[nvbench-compare]: https://github.com/NVIDIA/nvbench/blob/main/docs/nvbench_compare_robust.md

The model section describes algorithmic FLOPs and estimated traffic. The
utilization section uses [CUPTI][cupti] hardware counters collected in a
separate replay. Read-only loads over twice the L2 capacity evict cached data
before each counter pass. [Cache discard][cache-discard] then removes only
private scratch lines so they cannot contribute dirty writebacks. GPU timing
remains unprofiled.

DRAM reads, writes, bandwidth, and utilization come from `dram__bytes` counters.
FP32 work counts executed ADD and MUL instructions plus twice the executed FMA
instructions; it excludes Tensor Core and transcendental operations. Sustained
peak rates come from the counter metrics. FP64 uses the same counting rule.
FP16 reports ADD/MUL/FMA instruction throughput utilization. Tensor reports
active pipeline cycles. Neither infers FLOPs from types or matrix dimensions.
Integer reports predicated thread instruction counts and rates, with ALU
pipeline utilization. This includes address and index arithmetic and does not
isolate INT32 application work.

Global ATOM, global RED, and shared atomic counts use warp instruction counters.
L2 ATOM and RED sectors, sector rates, and sustained-peak percentages appear
separately. Shared atomic utilization measures instruction throughput. These
percentages measure throughput, not contention latency. An unused `atomicAdd`
result can compile to RED. Inactive compute meters are omitted.
L2 and DRAM counters include [traffic from other GPU contexts][ncu-shared].
Run without concurrent GPU work to attribute that traffic to the app.
Counter collection failures report unavailable metrics without a modeled
fallback. Occupancy also shows the compiled kernel's theoretical limit.

For instruction-level analysis, use [`ncu`][ncu]. Profiling under Nsight
disables the embedded collector because Nsight owns the CUPTI session.

[ncu]: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
[ncu-shared]: https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/known-issues.html
[cupti]: https://docs.nvidia.com/cupti/main/main.html#cupti-range-profiling-api
[cache-discard]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-discard

Fix or check C++ and CUDA sources. Python is linted and formatted with ruff:

```sh
mise run fmt
mise run lint
prek --config .config/prek.toml run --all-files
```

[`clang-tidy`][clang-tidy] runs inside the CUDA container against
`build/docker`. [`prek`][prek] runs `mise run lint` before commits and treats
every diagnostic as an error. `mise install` installs its Git hook after the
tools.

[clang-tidy]: https://clang.llvm.org/extra/clang-tidy/
[prek]: https://prek.j178.dev/

## Notebooks

[JupyterLab][jupyterlab] uses an isolated [uv][uv] environment under
`src/notebooks/`. No notebooks are required or shipped. Group local notebooks
by topic within that directory.

```sh
mise run nb
```

Create a Python notebook from the launcher. To use text notebooks, open a MyST
Markdown file through [Jupytext][jupytext]. Saving a Markdown notebook writes
MyST Markdown without creating an `.ipynb`. Its outputs live in the browser
session and are lost when reopened. Run the cells again to regenerate them.

Playground contents are gitignored except for the environment files. To share a
notebook, add its exact path as a `!` exception at the end of `.gitignore`, then
add the file to Git. Keep required companion files as explicit exceptions too.

Markdown notebooks open with code inputs collapsed and outputs visible.
Input visibility stays in the browser; saving does not add per-cell visibility
metadata. Markdown cells render automatically after edits.

Edit the Markdown in Neovim. The local [reload plugin][reload-plugin] checks
open Markdown notebooks every second and loads external saves automatically.
It preserves the running kernel and outputs for unchanged code cells. Changed
code cells lose their old outputs; run them again to update results.

Reload pauses while the kernel runs or the browser has unsaved source edits.
Save or discard browser edits before switching editors. Browser autosave is
disabled. **File → Reload from Disk** remains available for manual reload.

[jupyterlab]: https://jupyterlab.readthedocs.io/en/stable/
[uv]: https://docs.astral.sh/uv/
[jupytext]: https://jupytext.org/getting-started/text-notebooks/
[reload-plugin]: ../.config/jupyter/autoreload.js

## macOS: CLion Intellisense

CLion runs on the Mac without `nvcc`. Point it at the Docker toolchain so it
indexes against CUDA headers inside the image:

1. Run `mise run dc:up`.
2. Add a Docker toolchain using the configured `CONTAINER_NAME` image.
3. Make it the default toolchain, or pick it in the `clion` profile.

Builds and deployments still use the mise tasks. See the [CLion Docker toolchain
documentation][clion-docker].

[clion-docker]: https://www.jetbrains.com/help/clion/clion-toolchains-in-docker.html

### Shared CMake profile

[`.idea/cmake.xml`][cmake-xml] is the one tracked `.idea` file: a [shared
profile][clion-share] named `clion` that generates into `build/clion`, beside
`build/docker` and `build/local`. A root [global ignore][git-ignore] drops
`.idea/`, so `.gitignore` re-includes this file alone.

Enable it under **Settings | Build, Execution, Deployment | CMake** and delete
the local `Debug` profile; a local profile of the same name hides the shared
one. Toolchain, generator, and build directory of *local* profiles stay in
`workspace.xml`, which is why only the shared profile can pin the tree.

[cmake-xml]: ../.idea/cmake.xml
[clion-share]: https://www.jetbrains.com/help/clion/cmake-profile.html#share-profiles
[git-ignore]: https://git-scm.com/docs/gitignore

### Format on save

nvim formats `.cu` / `.cuh` / `.cpp` / headers on save with `clang-format` and
this repo's [`.clang-format`][clang-format-config]. CLion is configured the same
way via `.idea` (untracked below, so redo this on a new machine):

[clang-format-config]: ../.clang-format

- `.idea/codeStyles/Project.xml`, `.idea/editor.xml` — ClangFormat is the
  formatting engine, so `.clang-format` wins over CLion's own code style.
- `.idea/workspace.xml` — `FormatOnSaveOptions` reformats on save, scoped to the
  `C/C++` and `C/C++ Header` file types. CMake and Markdown stay untouched.

CLion formats with the clang-format built into its bundled clangd, not the
`clang-format` pinned in `.config/mise/config.toml`. For an exact match, set
**Settings | Editor | Code Style | C/C++ | General** to use an external
clang-format at the path from `mise which clang-format`.
