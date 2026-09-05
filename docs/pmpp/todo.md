# TODO

## [`../../src/pmpp/ch03/blur.cu`][blur]

[blur]: ../../src/pmpp/ch03/blur.cu

Ch03 kernel: one thread per pixel, full 2D window from global memory. Correct
teaching code. Slow box blur.

Bottleneck is redundant DRAM traffic, not occupancy. Adjacent threads almost
fully overlap the window. Packed RGB (3 bytes) also hurts coalescing.

Book path (same convolution pattern, not this file):

- Ch 5–6: tiling, coalescing, coarsening
- Ch 7: constant-memory filter, tiled convolution + halo, cache for halo
- Ch 8: shared memory / register tiling for neighborhood updates

After that, extra algorithm changes the book does not require:

- Separable H then V: O(R) instead of O(R²)
- Prefix / integral image: O(1) per pixel after a scan (ch 11)
- `dim3(32, 8)` (or a row kernel) vs `16×16` for coalescing
- Planar RGB or RGBX if load width matters

Revisit `blur.cu` once ch 7 (and ideally 8) is done. Timing and Nsight:
[`../bench/readme.md`][benchmarking].

[benchmarking]: ../bench/readme.md
