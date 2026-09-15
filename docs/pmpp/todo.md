# TODO

## Blur

[`../../src/pmpp/ch03/blur.cu`][blur]. Ch03 kernel: one thread per pixel, full
2D window from global memory. Correct teaching code. Slow box blur.

Bottleneck is redundant DRAM traffic, not occupancy. Adjacent threads almost
fully overlap the window. Packed RGB (3 bytes) also hurts coalescing.

Book path (same convolution pattern, not this file):

- Ch 5–6: tiling, coalescing, coarsening
- Ch 7: constant-memory filter, tiled convolution + halo, cache for halo
- Ch 8: shared memory / register tiling for neighborhood updates

After that, extra algorithm changes the book does not require:

- Separable H then V: $O(R)$ instead of $O(R^2)$
- Prefix / integral image: $O(1)$ per pixel after a scan (ch 11)
- `dim3(32, 8)` (or a row kernel) vs $16 \times 16$ for coalescing
- Planar RGB or RGBX if load width matters

Revisit `blur.cu` once ch 7 (and ideally 8) is done. Timing and Nsight:
[`../bench/readme.md`][benchmarking].

[blur]: ../../src/pmpp/ch03/blur.cu
[benchmarking]: ../bench/readme.md

## Matmul

[`../../src/pmpp/ch03/matmul.kernels.cu`][matmul]. Ch03 `cell`: one thread per
$C$ element, inner product over $k$ from global memory. Coalesced along $B$.
Slow GEMM.

[`../../src/pmpp/ch05/matmul.tile.cpp`][matmul-tile]. Ch5 Fig. 5.9:
$16\times 16$ shared tiles, one $C$ element per thread.

[matmul]: ../../src/pmpp/ch03/matmul.kernels.cu
[matmul-tile]: ../../src/pmpp/ch05/matmul.tile.cpp

Bottleneck for `cell` is reuse, not the FMA. A warp shares a row of $A$
($A[\mathrm{row}, i]$) and walks $B$ coalesced. Ampere L1/L2 already capture
much of that, so Fig. 5.9 is not $T\times$ wall-clock for tile width $T$.

Ch 5 (book vs timer):

- Ex 5.2 / 5.5 / 5.8: algorithmic global loads drop by $T$. Access count, not
  runtime.
- Expect about $1.5\times$ vs coalesced `cell` and a few TFLOP/s, not $T\times$
  and not peak FP32.
- Same step as [Siboehm][gemm-worklog] kernel 2 (coalesced GMEM, $\sim 1986$
  GFLOP/s) to kernel 3 (SMEM tile, $\sim 2980$ GFLOP/s) on RTX A6000.
- Other Fig. 5.9-style runs: [Holt][holt-tiled] $\sim 2$–$3\times$ on A40;
  student dumps often $\sim 1.1$–$1.8\times$.
- nvbench `BWUtil` $\gg 100$% counts $2mnk$ float loads against DRAM peak. Use
  Nsight `dram__bytes` for DRAM.

To raise $t_{\mathrm{cell}}/t_{\mathrm{tile}}$: grow $m,n,k$ (especially $k$)
so `cell` misses L2; try `kTileWidth` $32$. Smaller matrices make the ratio
worse. Still do not expect $T\times$ time.

Book path:

- Ch 6.3: 1D thread coarsening on the tiled kernel (Siboehm kernel 4)
- Later chapters reuse coarsening and tiling on other patterns, not the rest of
  that SGEMM list
- After 6.3, [Siboehm][gemm-worklog] kernels 5–11 and [CUTLASS efficient
  GEMM][cutlass-gemm] for 2D register tiles, vector loads, SMEM swizzle, warp
  tiles, double buffering. Ch 16 is conv-as-GEMM + cuDNN, not handwritten SGEMM
- Compare to the CUTLASS oracle, not only `cell`

Revisit `matmul.kernels.cu` after ch 6.3. Timing and Nsight:
[`../bench/readme.md`][benchmarking].

See also:

- [Anatomy of a CUDA GEMM][gemm-anatomy]
- [How to Optimize a CUDA Matmul Kernel][gemm-worklog]

[cutlass-gemm]:
  <https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html>
[gemm-anatomy]:
  <https://medium.com/@emmanuelalo52/anatomy-of-a-cuda-gemm-from-naive-kernels-to-outperforming-cublas-on-blackwell-c394b04b5995>
[gemm-worklog]: https://siboehm.com/articles/22/CUDA-MMM
[holt-tiled]: https://andreasholt.com/posts/shared-tiled-matmul/
