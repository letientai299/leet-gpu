# Benchmarking and profiling CUDA kernels

PMPP teaches *what* to look at (occupancy, coalescing, traffic). It does not
teach a timing harness or Nsight. Use this note when measuring kernels in
`apps/`.

CUDA has both **micro** (one kernel) and **macro** (whole app) tools. They are
not one `pprof` binary.

## Map from Go

| Go | CUDA |
| --- | --- |
| `go test -bench` | Timer around **one kernel**: `cudaEvent`, warmup, many launches. That is the microbenchmark. There is no `testing.B`. |
| `pprof` CPU (who is hot) | **Nsight Systems** (`nsys`): CPU + GPU **timeline**. Kernels, memcpy, idle GPU, launch gaps. |
| One hot function in pprof | **Nsight Compute** (`ncu`): **one kernel**, occupancy, coalescing, roofline, warp stalls. |
| `go tool trace` | Closest is **Nsight Systems**. |
| Heap/alloc pprof | No device-heap analog. Use **compute-sanitizer**, not a profiler. |

`nvprof` is gone. Use `nsys` + `ncu`.

**Macro** = host, copies, kernel, overlap, PNG I/O. **Micro** = one kernel’s
hardware counters. CUDA needs both; pprof often collapses them into “this
function is hot.”

Libraries exist (NVBench, Google Benchmark + CUDA). For this repo, events +
`nsys` / `ncu` are enough.

## Order of work

1. **Harness first.** Do not open a GUI until kernel-only time is stable.
2. **`nsys`.** Is the GPU busy, or are you timing memcpy / launch / disk?
3. **`ncu`.** Why is *this* kernel slow?

For image apps (`grayscale`, `blur`), wall-clock of the binary includes PNG
decode/encode. Time the kernel, not `main`.

## Timing harness

Use CUDA events, not host clocks across an async launch:

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

kernel<<<grid, block>>>(...); // warmup: JIT + caches
cudaDeviceSynchronize();

cudaEventRecord(start);
for (int i = 0; i < N; i++) {
  kernel<<<grid, block>>>(...);
}
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms = 0;
cudaEventElapsedTime(&ms, start, stop);
```

Report **median** (or min) of several runs, not a single shot. Time **H2D /
kernel / D2H** as three ranges. `cudaEventElapsedTime` is milliseconds.

Pitfalls:

- First launch can include **module load / JIT**. Warmup once, then measure.
- Launch is async. Host `clock` without sync measures queueing, not GPU work.
- Tiny kernels: launch overhead dominates. Raise work or batch launches.
- Changing `BLUR_SIZE` without kernel-only time often just measures PNG size.

Docs: [CUDA events](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html).

## Nsight Systems (macro)

[Nsight Systems](https://developer.nvidia.com/nsight-systems) —
[user guide](https://docs.nvidia.com/nsight-systems/).

```sh
nsys profile -o out ./bin/pmpp/ch03/blur -i in.png -o out.png
```

Open `out.nsys-rep` in `nsys-ui`, or `nsys stats`. Look for: memcpy vs kernel,
GPU idle, many tiny kernels.

**NVTX** names ranges so the timeline is readable (like pprof labels):
[NVTX](https://nvidia.github.io/NVTX/). Wrap upload, kernel, download, encode.

## Nsight Compute (micro)

[Nsight Compute](https://developer.nvidia.com/nsight-compute) —
[profiling guide](https://docs.nvidia.com/nsight-compute/).

```sh
ncu --kernel-name blur_kernel --launch-skip 1 --launch-count 5 -o blur \
  ./bin/pmpp/ch03/blur -i in.png -o out.png
```

`--set full` is heavy; start with default / Speed of Light. Read: roofline,
DRAM throughput, occupancy, uncoalesced access, warp stall reasons.

Replay/kernel-id flags matter: `ncu` can rerun the kernel many times. Skip
warmup launches. On some shared GPUs, counters need extra permissions
(`ERR_NVGPUCTRPERM`).

## What PMPP will still not cover

Ch 4 occupancy, ch 6 coalescing / bottleneck checklist, ch 22 bandwidth vs
compute. No harness, no `nsys`/`ncu` tutorial.

When optimizing [`apps/pmpp/ch03/blur.cu`](../../apps/pmpp/ch03/blur.cu), see
[`docs/pmpp/todo.md`](../pmpp/todo.md). Measure kernel-only time *before*
tiling so later chapters have a baseline.

## Resources

- NVIDIA CUDA Developer Tools videos: Intro to Nsight Systems, Intro to Nsight
  Compute (from the product pages above).
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
  — measure first, then memory and occupancy.
- CUPTI sits under `nsys`/`ncu`; skip it until the CLIs are familiar.
