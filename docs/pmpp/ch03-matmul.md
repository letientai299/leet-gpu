# Matmul benchmark analysis

H100 80GB HBM3, 132 SMs, SM clock locked at 1980 MHz (P0), `Clock Scaling`
100% on every row below. [NVBench][nvbench] settings: 5 warmup runs, min 20
samples, 1 s batch target, throttle threshold 0.9.

$$
\mathrm{GFLOPs/s} = \frac{2MNK}{t_{\mathrm{GPU}}\times 10^9}
$$

- Cold: isolated launches, one sync per launch. Includes launch overhead.
- Batch: many launches, total GPU time / count. Launch cost amortized.
- Logical BW is NVBench `GlobalMem BW`, not DRAM. The harness declares
  $2MNK$ float reads (one $A$ and one $B$ per FMA) and $MN$ float writes, then
  NVBench computes $(2MNK+MN)\times 4 / t_{\mathrm{GPU}}$. That is the naive
  kernel's request stream if every load hit global memory. Cache can satisfy
  those loads, so the column can exceed HBM peak (3.35 TB/s). Cell ~19 TB/s on
  $2048^3$ means L1/L2 reuse, not 19 TB/s of DRAM. Row/col look like tens of
  GB/s because they are slow. Use [ncu][ncu] for real sectors and occupancy.

## Summary

| Kernel | Threads | Peak batch GFLOPs/s |                            Best shape |      Aligned vs ragged |
| ------ | ------: | ------------------: | ------------------------------------: | ---------------------: |
| cell   |    $MN$ |                4803 |                     occupied $2048^3$ |                    ±2% |
| col    |     $N$ |                87.3 |        wide $128{\times}2048$, $K=64$ |                    ±2% |
| row    |     $M$ |                15.6 | tall ragged $2053{\times}127$, $K=67$ | ragged 1.3–1.4× faster |

- Cell wins every shape by 2–3 orders of magnitude: $MN$ threads and coalesced
  $B$/$C$.
- Row vs col follows occupancy against coalescing. Tall ($M\gg N$): row. Square
  and wide ($N\ge M$): col.
- Row is the only kernel sensitive to alignment. Power-of-two $N$ gives its
  strided walk a power-of-two stride ($N=128$ → 512 B, $N=2048$ → 8 KiB), which
  concentrates the warp on few cache sets and memory partitions. Ragged $N$
  spreads it.
- Row and col GFLOPs/s are flat in $K$; they are occupancy/latency bound. Cell
  rises with $K$ (more reuse per launch) and saturates near 4.2 TFLOPs/s at
  $MN=262144$.

## Occupied squares

Every kernel gets thousands of threads here, so these are the honest rates.

| Case    |    Shape | Kernel |   Cold GPU | Noise | Cold GFLOPs/s |  Batch GPU | Batch GFLOPs/s |  Logical BW |
| ------- | -------: | ------ | ---------: | ----: | ------------: | ---------: | -------------: | ----------: |
| aligned | $2048^3$ | cell   |   3.594 ms | 0.04% |        4779.8 |   3.577 ms |         4803.4 | 19.124 TB/s |
| aligned | $2048^3$ | row    |    1.554 s | 0.02% |        11.055 |    1.554 s |         11.056 | 44.232 GB/s |
| aligned | $2048^3$ | col    | 687.106 ms | 0.01% |        25.003 | 686.528 ms |         25.024 | 100.04 GB/s |
| ragged  | $2053^3$ | cell   |   3.665 ms | 0.10% |        4721.9 |   3.656 ms |         4733.5 | 18.892 TB/s |
| ragged  | $2053^3$ | row    |    1.172 s | 0.07% |        14.765 |    1.172 s |         14.768 | 59.075 GB/s |
| ragged  | $2053^3$ | col    | 685.346 ms | 0.00% |        25.251 | 685.199 ms |         25.257 | 101.03 GB/s |
| aligned | $4096^3$ | cell   |  29.880 ms | 0.03% |        4599.7 |  29.867 ms |         4601.7 | 18.401 TB/s |
| ragged  | $4092^3$ | cell   |  29.330 ms | 0.09% |        4672.2 |  29.336 ms |         4671.3 | 18.691 TB/s |

$4092^3$ is [Boehm's][boemm] size. His kernel 1 (uncoalesced naive) reached 309
GFLOPs/s and kernel 2 (coalesced naive) 1986 GFLOPs/s on an A6000. Our cell
kernel already maps `threadIdx.x` to the column, so it is a kernel-2 mapping,
and this H100 is a much larger part: 4671 GFLOPs/s. Still no tiling, no shared
memory, no tensor cores, so cuBLAS-class rates stay far away.

## Aspect suite

$MN$ fixed near 262144 so only the mapping changes. Ragged rows use no
dimension that is a multiple of 16 or 32, so every kernel takes its
bounds-check path with a partial warp.

### cell

| Shape         |      $M{\times}N$ |  $K$ |   Cold GPU | Noise | Cold GFLOPs/s |  Batch GPU | Batch GFLOPs/s |  Logical BW |
| ------------- | ----------------: | ---: | ---------: | ----: | ------------: | ---------: | -------------: | ----------: |
| Tall          | $2048{\times}128$ |   64 |  12.465 us | 1.28% |        2691.8 |  10.868 us |         3087.4 | 10.851 TB/s |
| Tall ragged   | $2053{\times}127$ |   67 |  13.230 us | 1.53% |        2640.9 |  11.146 us |         3134.7 | 10.642 TB/s |
| Tall          | $2048{\times}128$ |  512 |  72.023 us | 0.48% |        3727.1 |  68.088 us |         3942.5 | 14.923 TB/s |
| Tall ragged   | $2053{\times}127$ |  509 |  77.329 us | 0.62% |        3432.4 |  66.940 us |         3965.1 | 13.743 TB/s |
| Tall          | $2048{\times}128$ | 2048 | 279.768 us | 0.43% |          3838 | 253.999 us |         4227.3 | 15.356 TB/s |
| Tall ragged   | $2053{\times}127$ | 2053 | 296.859 us | 0.39% |        3606.3 | 255.742 us |         4186.1 | 14.429 TB/s |
| Square        |  $512{\times}512$ |   64 |  12.474 us | 1.54% |        2689.9 |  10.732 us |         3126.5 | 10.844 TB/s |
| Square ragged |  $515{\times}509$ |   67 |  12.902 us | 4.24% |        2722.4 |  11.109 us |           3162 | 10.971 TB/s |
| Square        |  $512{\times}512$ |  512 |  69.995 us | 0.66% |          3835 |  67.280 us |         3989.8 | 15.355 TB/s |
| Square ragged |  $515{\times}509$ |  509 |  70.055 us | 0.85% |        3809.2 |  66.007 us |         4042.8 | 15.252 TB/s |
| Square        |  $512{\times}512$ | 2048 | 266.314 us | 0.42% |        4031.9 | 257.914 us |         4163.2 | 16.131 TB/s |
| Square ragged |  $515{\times}509$ | 2053 | 271.377 us | 0.57% |        3966.2 | 254.031 us |           4237 | 15.869 TB/s |
| Wide          | $128{\times}2048$ |   64 |  12.799 us | 4.29% |        2621.6 |  10.859 us |         3089.9 | 10.568 TB/s |
| Wide ragged   | $127{\times}2053$ |   67 |  13.059 us | 1.42% |        2675.4 |  11.041 us |         3164.3 | 10.781 TB/s |
| Wide          | $128{\times}2048$ |  512 |  71.260 us | 2.39% |          3767 |  67.098 us |         4000.6 | 15.083 TB/s |
| Wide ragged   | $127{\times}2053$ |  509 |  66.060 us | 1.02% |        4017.9 |  64.880 us |           4091 | 16.087 TB/s |
| Wide          | $128{\times}2048$ | 2048 | 267.211 us | 2.34% |        4018.3 | 259.313 us |         4140.7 | 16.077 TB/s |
| Wide ragged   | $127{\times}2053$ | 2053 | 251.627 us | 0.40% |        4254.6 | 253.433 us |         4224.2 | 17.022 TB/s |

Aspect ratio barely matters: $MN$ threads and a coalesced warp either way.
$K$ matters, and cold trails batch by ~15% at $K=64$ but ~2% at $K=2048$ —
launch overhead against real work.

### row

| Shape         |      $M{\times}N$ |  $K$ |   Cold GPU | Noise | Cold GFLOPs/s |  Batch GPU | Batch GFLOPs/s |  Logical BW |
| ------------- | ----------------: | ---: | ---------: | ----: | ------------: | ---------: | -------------: | ----------: |
| Tall          | $2048{\times}128$ |   64 |   3.107 ms | 0.21% |        10.801 |   3.058 ms |         10.973 | 43.542 GB/s |
| Tall ragged   | $2053{\times}127$ |   67 |   2.254 ms | 0.01% |        15.498 |   2.243 ms |         15.577 | 62.455 GB/s |
| Tall          | $2048{\times}128$ |  512 |  24.180 ms | 0.31% |        11.101 |  23.620 ms |         11.365 | 44.449 GB/s |
| Tall ragged   | $2053{\times}127$ |  509 |  17.983 ms | 0.08% |         14.76 |  17.866 ms |         14.857 | 59.097 GB/s |
| Tall          | $2048{\times}128$ | 2048 |  96.702 ms | 0.06% |        11.104 |  94.413 ms |         11.373 | 44.425 GB/s |
| Tall ragged   | $2053{\times}127$ | 2053 |  72.768 ms | 0.08% |        14.712 |  72.394 ms |         14.788 | 58.863 GB/s |
| Square        |  $512{\times}512$ |   64 |  12.286 ms | 0.13% |         2.731 |  11.869 ms |          2.827 | 11.009 GB/s |
| Square ragged |  $515{\times}509$ |   67 |   8.994 ms | 0.01% |        3.9055 |   8.976 ms |         3.9131 | 15.739 GB/s |
| Square        |  $512{\times}512$ |  512 |  97.083 ms | 0.07% |         2.765 |  93.585 ms |         2.8684 | 11.071 GB/s |
| Square ragged |  $515{\times}509$ |  509 |  71.866 ms | 0.03% |        3.7132 |  71.793 ms |          3.717 | 14.867 GB/s |
| Square        |  $512{\times}512$ | 2048 | 387.349 ms | 0.06% |         2.772 | 373.247 ms |         2.8768 | 11.091 GB/s |
| Square ragged |  $515{\times}509$ | 2053 | 289.576 ms | 0.05% |        3.7169 | 289.112 ms |         3.7229 | 14.871 GB/s |
| Wide          | $128{\times}2048$ |   64 |  27.062 ms | 0.03% |        1.2399 |  25.404 ms |         1.3208 |  4.998 GB/s |
| Wide ragged   | $127{\times}2053$ |   67 |  21.221 ms | 0.11% |        1.6464 |  19.476 ms |         1.7939 |  6.635 GB/s |
| Wide          | $128{\times}2048$ |  512 | 332.579 ms | 0.06% |       0.80713 | 319.506 ms |        0.84016 |  3.232 GB/s |
| Wide ragged   | $127{\times}2053$ |  509 | 224.323 ms | 0.14% |        1.1832 | 202.421 ms |         1.3112 |  4.738 GB/s |
| Wide          | $128{\times}2048$ | 2048 |    1.330 s | 0.03% |       0.80715 |    1.278 s |        0.83986 |  3.229 GB/s |
| Wide ragged   | $127{\times}2053$ | 2053 | 909.930 ms | 0.06% |        1.1765 | 819.494 ms |         1.3064 |  4.707 GB/s |

Row scales with $M$ (thread count) and degrades as $N$ grows (more serial work
per thread, wider stride). Every ragged row beats its aligned twin by
1.3–1.5×, the power-of-two stride penalty.

### col

| Shape         |      $M{\times}N$ |  $K$ |   Cold GPU | Noise | Cold GFLOPs/s |  Batch GPU | Batch GFLOPs/s |  Logical BW |
| ------------- | ----------------: | ---: | ---------: | ----: | ------------: | ---------: | -------------: | ----------: |
| Tall          | $2048{\times}128$ |   64 |   7.568 ms | 0.07% |        4.4338 |   5.809 ms |         5.7766 | 17.874 GB/s |
| Tall ragged   | $2053{\times}127$ |   67 |   8.077 ms | 0.05% |        4.3258 |   6.255 ms |         5.5858 | 17.432 GB/s |
| Tall          | $2048{\times}128$ |  512 | 175.322 ms | 0.05% |        1.5311 |  77.363 ms |         3.4698 |  6.130 GB/s |
| Tall ragged   | $2053{\times}127$ |  509 | 161.289 ms | 0.09% |        1.6456 | 146.217 ms |         1.8153 |  6.589 GB/s |
| Tall          | $2048{\times}128$ | 2048 | 705.626 ms | 0.03% |        1.5217 | 651.079 ms |         1.6492 |  6.088 GB/s |
| Tall ragged   | $2053{\times}127$ | 2053 | 708.608 ms | 0.01% |        1.5108 | 657.229 ms |         1.6289 |  6.045 GB/s |
| Square        |  $512{\times}512$ |   64 |   1.872 ms | 0.14% |         17.92 |   1.509 ms |         22.229 | 72.239 GB/s |
| Square ragged |  $515{\times}509$ |   67 |   2.006 ms | 0.17% |        17.511 |   1.620 ms |         21.682 | 70.567 GB/s |
| Square        |  $512{\times}512$ |  512 |  43.912 ms | 0.08% |         6.113 |  41.134 ms |         6.5259 | 24.476 GB/s |
| Square ragged |  $515{\times}509$ |  509 |  42.987 ms | 0.03% |        6.2077 |  41.325 ms |         6.4574 | 24.855 GB/s |
| Square        |  $512{\times}512$ | 2048 | 175.601 ms | 0.06% |        6.1147 | 164.289 ms |         6.5357 | 24.465 GB/s |
| Square ragged |  $515{\times}509$ | 2053 | 177.830 ms | 0.04% |        6.0526 | 167.101 ms |         6.4412 | 24.216 GB/s |
| Wide          | $128{\times}2048$ |   64 | 481.062 us | 0.32% |        69.751 | 384.456 us |         87.278 | 281.18 GB/s |
| Wide ragged   | $127{\times}2053$ |   67 | 495.939 us | 0.26% |        70.448 | 406.124 us |         86.028 | 283.90 GB/s |
| Wide          | $128{\times}2048$ |  512 |  10.843 ms | 0.03% |        24.756 |  10.540 ms |         25.468 | 99.121 GB/s |
| Wide ragged   | $127{\times}2053$ |  509 |  10.603 ms | 0.02% |        25.033 |  10.492 ms |         25.297 | 100.23 GB/s |
| Wide          | $128{\times}2048$ | 2048 |  43.349 ms | 0.02% |        24.769 |  42.105 ms |         25.501 | 99.102 GB/s |
| Wide ragged   | $127{\times}2053$ | 2053 |  42.810 ms | 0.01% |        25.007 |  42.352 ms |         25.278 | 100.05 GB/s |

Col scales with $N$ and is alignment-insensitive: its warp already reads
consecutive $B$/$C$, so ragged only trims a partial warp. Aligned and ragged
agree within 2% on every row except tall $K=512$, where the aligned cold pass
hit an outlier (cold 175 ms vs batch 77 ms).

## Ragged correctness shape

$M=67$, $N=33$, $K=50$: the PMPP bounds-check case and `matmul.bench`'s default.
About 29 KiB total, 15 cell blocks against 132 SMs. Launch-bound, so treat it
as a smoke test, not a rate.

| Kernel |  Cold mean | Cold GFLOPs/s | Batch mean | Batch GFLOPs/s |
| ------ | ---------: | ------------: | ---------: | -------------: |
| cell   |   8.462 us |        26.128 |   4.772 us |         46.330 |
| row    | 154.614 us |         1.430 | 136.814 us |          1.616 |
| col    | 207.930 us |         1.063 | 153.245 us |          1.443 |

## Ex 3.1c

[Guess][ch03-ex]: row > col > cell, because more parallel writers to $C$ looked
like more memory contention.

Wrong on both counts. Cell is fastest everywhere, and there is no CPU-style
cache-line lock: cost is a warp of consecutive threads touching scattered
addresses. One thread walking a row later is locality for that thread, not
coalescing for the warp.

- Row beats col only when $M\gg N$ (tall), on thread count.
- Column-major storage would flip which coarsened kernel coalesces.
- Row and col write `C` inside the $K$ loop. Check PTX/SASS for whether the
  compiler keeps a register accumulator.
- Later chapters add tiling and shared memory: [`todo.md`][matmul-todo].

## Commands

`matmul.bench`: one host fill, one device `{A,B,C}` per GPU, all three kernels.
Lock clocks in separate calls; the NVBench parser exits after applying them.

```sh
sudo matmul.bench --lock-gpu-clocks max
matmul.bench --height 2048 --width 128 --k 512 --jsonbin matmul-tall.jsonbin \
  --markdown matmul-tall.md
sudo matmul.bench --lock-gpu-clocks reset
```

`matmul --kernel cell|row|col`: one kernel; axes form a Cartesian product.

```sh
matmul --kernel row --bench \
  --axis 'Height=[128,512,2048]' \
  --axis 'Width=[128,512,2048]' \
  --axis 'K=[64,512,2048]'
```

Shapes above: aligned $\{2048{\times}128, 512{\times}512, 128{\times}2048\}$
with $K\in\{64,512,2048\}$, ragged $\{2053{\times}127, 515{\times}509,
127{\times}2053\}$ with $K\in\{67,509,2053\}$, plus $2048^3$, $2053^3$,
$4092^3$, and $4096^3$.

## References

- [ ] [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance][boemm]

[boemm]: https://siboehm.com/articles/22/CUDA-MMM
[ch03-ex]: ./ch03-ex.md#ex-31
[matmul-todo]: ./todo.md#matmul
[ncu]: https://docs.nvidia.com/nsight-compute/
[nvbench]: https://github.com/NVIDIA/nvbench
