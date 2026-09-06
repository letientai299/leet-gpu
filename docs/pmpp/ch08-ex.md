# Chapter 8 exercises — Stencil

Back to the [reading checklist][reading-checklist].

[reading-checklist]: ./readme.md#8-stencil

Questions transcribed from the book.

## Ex 8.1

Consider a 3D stencil computation on a grid of size $120 \times 120 \times 120$,
including boundary cells.

- a. What is the number of output grid points that is computed during each
  stencil sweep?
- b. For the basic kernel in Fig. 8.6, what is the number of thread blocks that
  are needed, assuming a block size of $8 \times 8 \times 8$?
- c. For the kernel with shared memory tiling in Fig. 8.8, what is the number
  of thread blocks that are needed, assuming a block size of
  $8 \times 8 \times 8$?
- d. For the kernel with shared memory tiling and thread coarsening in
  Fig. 8.10, what is the number of thread blocks that are needed, assuming a
  block size of $32 \times 32$?

**Answer:**

## Ex 8.2

Consider an implementation of a seven-point (3D) stencil with shared memory
tiling and thread coarsening applied. The implementation is similar to those in
Figs. 8.10 and 8.12, except that the tiles are not perfect cubes. Instead, a
thread block size of $32 \times 32$ is used as well as a coarsening factor of
$16$ (i.e., each thread block processes $16$ consecutive output planes in the
$z$ dimension).

- a. What is the size of the input tile (in number of elements) that the thread
  block loads throughout its lifetime?
- b. What is the size of the output tile (in number of elements) that the
  thread block processes throughout its lifetime?
- c. What is the floating point to global memory access ratio (in
  $\mathrm{OP/B}$) of the kernel?
- d. How much shared memory (in bytes) is needed by each thread block if
  register tiling is not used, as in Fig. 8.10?
- e. How much shared memory (in bytes) is needed by each thread block if
  register tiling is used, as in Fig. 8.12?

**Answer:**
