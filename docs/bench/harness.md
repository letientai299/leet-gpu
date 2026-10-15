# Correctness and benchmark harnesses

Run the harness regressions with the local CUDA toolkit:

```sh
mise run test-harness
mise run test-harness -- --gpu
```

The first command runs host and script tests without a GPU. The second also
runs GPU tests on every visible device. Set `CUDA_VISIBLE_DEVICES` to restrict
that test run. Tests live in [`src/libs/tests`][tests].

## Capture a benchmark

Set `APP` to a built application that supports `--bench`. Run on the GPU host:

```sh
mkdir -p .ai/kernel-bench
SSH_HOST=local BENCH_GPU=0 mise run bench -- "$APP" --bench \
  --jsonbin .ai/kernel-bench/run.json 2> .ai/kernel-bench/run.log
mise run bench-compare -- .ai/kernel-bench/run.json baseline candidate
```

Use the registered benchmark names for `baseline` and `candidate`. The
comparison requires both kernels in the same run. Keep the JSON and its
binary sample files together. The task preserves relative sample paths. To
compare separate apps, use `mise cmp <baseline> <app>...`.

Omit `--jsonbin` for a vertical text report. Host diagnostics, GPU settings,
the command, and the executable hash go to stderr, leaving `--json stdout`
parseable. Save stderr beside the results for reproduction.

For a short timing experiment, append `--min-time 0.1 --timeout 5
--no-batch`. Use repeated longer runs before interpreting small changes.
NVBench options pass through to the [measurement runner][runner].

## GPU settings

`BENCH_GPU` selects an NVIDIA management index or UUID. The task resolves it to
a UUID before setting `CUDA_VISIBLE_DEVICES`; the selected GPU appears as CUDA
device 0 inside the application. [CUDA documents this remapping][visibility].

`BENCH_CLOCK=off` is the default and preserves existing clock settings. To
lock a GPU whose clocks were previously unlocked:

```sh
SSH_HOST=local BENCH_GPU=0 BENCH_CLOCK=auto BENCH_RESTORE_CLOCK=off \
  mise run bench -- "$APP" --bench
```

For an existing lock, set `BENCH_RESTORE_CLOCK` to its previous MHz value or
`min,max` range. The task cannot discover that range. `BENCH_CLOCK=auto`
selects a supported clock using `BENCH_CLOCK_FRACTION`; a numeric value
requests a specific MHz clock. Locking needs permission to run
[`nvidia-smi`][smi] write operations. A failed lock fails the run.

The task refuses active compute processes and waits for its temperature and
power thresholds. `BENCH_FORCE=1` permits a busy GPU; `BENCH_ALLOW_HOT=1`
permits a cooldown timeout. Both can reduce repeatability. Cooperating runs
in the same checkout share a GPU lock; external jobs can still start.
SIGINT and SIGTERM stop the child before restoring settings. SIGKILL and
machine failure cannot run cleanup traps.

## Correctness scope

Correctness runs poison outputs with NaNs before launching the candidate.
Unwritten outputs fail even when the reference value is zero. Comparisons
reject non-finite candidate and reference values and report mismatch count,
first mismatch, maximum absolute and relative errors, and RMS error over
finite pairs. Near-zero references can make relative error large; acceptance
uses absolute plus relative tolerance.

Every timed state checks its actual shape and selected device first.
GEMM uses the CUTLASS FP32 reference; 2D convolution uses NPP with explicit
padding and filter orientation. Both compare every output.

3D correctness mode compares every output against the CPU reference. Large
3D benchmark states compare deterministic reference samples and scan every
output for non-finite values. `Reference Samples` reports the limited
coverage. A finite error outside those samples can pass; use a full check
and Compute Sanitizer when validating a new kernel. Sampling keeps CPU
reference work out of benchmark preparation at full-volume scale.

Correctness and allocation happen before timing. Callbacks must enqueue work
on the supplied stream and overwrite every output. The harness does not
prove absence of out-of-bounds writes, races, or input mutation.

## Read the report

- Cold GPU time measures isolated executions; batch time measures repeated
  execution. Their cache and launch conditions differ.
- Median, quartiles, range, relative noise, sample count, and wall time expose
  variation. A timeout or noise warning needs investigation even if usable
  samples remain. Skipped states and insufficient samples return failure.
- Registers, local memory, shared memory, and theoretical active warps help
  explain resource limits. [Theoretical occupancy][occupancy] is a launch
  capacity estimate. After timing, one extra launch under the [CUPTI range
  profiler][cupti] measures achieved occupancy: Nsight Compute's
  `sm__warps_active.avg.pct_of_peak_sustained_active`. The Occupancy meter
  fills to achieved and shades up to theoretical; the gap comes from tail
  waves and uneven blocks. The row falls back to theoretical when counters
  are unavailable, such as under `ncu` or `nsys`, or without [profiling
  permission][counters].
- Model GMEM bytes, GB/s, and FLOP/B describe the callback's logical traffic
  model. Caches and reuse can make this differ from measured DRAM traffic.

[tests]: ../../src/libs/tests
[runner]: ../../src/libs/benchmark/runner.cu
[visibility]: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html
[smi]: https://docs.nvidia.com/deploy/nvidia-smi/
[occupancy]: https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
[cupti]: https://docs.nvidia.com/cupti/main/main.html#cupti-range-profiling-api
[counters]: https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters
