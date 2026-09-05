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

## Matmul

[`../../src/pmpp/ch03/matmul.cu`][matmul]. Ch03 kernel: one thread per $C$
element, inner product over $k$ from global memory. Correct teaching code.
Slow GEMM.

[matmul]: ../../src/pmpp/ch03/matmul.cu

Bottleneck is DRAM traffic and reuse, not the math. A warp shares a row of
$A$ (many threads reload the same $A[\mathrm{row}, i]$) while walking a
row of $B$ ($B[i, \mathrm{col}]$ is coalesced). No tile stays in shared
memory, so each inner step hits global again.

Book path:

- Ch 5: tiled multiply, coalescing, shared-memory tiles of $A$ and $B$
- Ch 6: more GEMM tuning (thread coarsening, avoiding shared-memory bank
  conflicts)
- Later: compare tiled kernel to the CUTLASS oracle, not only the naive one

Revisit `matmul.cu` once ch 5 (and ideally 6) is done. Timing and Nsight:
[`../bench/readme.md`][benchmarking].

[benchmarking]: ../bench/readme.md
