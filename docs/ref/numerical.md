# Numerical and scientific computing reading list

Covers the computations PMPP.4e parallelizes and the fields they come from.
Each section names the source chapters it supports; see the source tags in
[`readme.md`](./readme.md). See [`math.md`](./math.md) for the
underlying math and [`cuda.md`](./cuda.md) for the CUDA guides and libraries.

## Floating point

Supports PMPP.4e appendix A and every correctness check in this repository.

1. [What Every Computer Scientist Should Know About Floating-Point
   Arithmetic][goldberg]: representation, rounding, cancellation, and error
   analysis.
2. [Nick Higham's "What Is" articles][higham-what-is]: short explanations of
   condition numbers, backward error, fused multiply-add, mixed precision, and
   stochastic rounding.
3. Higham, _Accuracy and Stability of Numerical Algorithms_: the full treatment
   of rounding error analysis, including summation and matrix multiplication.
   SIAM; not free.
4. [Floating Point and IEEE 754][cuda-fp]: the CUDA-specific behavior.

[cuda-fp]: https://docs.nvidia.com/cuda/floating-point/
[goldberg]: https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
[higham-what-is]: https://nhigham.com/index-of-what-is-articles/

## Dense linear algebra and GEMM

Supports PMPP.4e ch. 3 to 6. Blocked matrix multiplication, data reuse, and the
roofline model are the same ideas on CPUs and GPUs.

1. [How to Optimize GEMM][how-to-gemm]: step-by-step CPU GEMM from naive loops
   to blocked, packed, and vectorized kernels.
2. [LAFF-On Programming for High Performance][pfhp]: the course version of the
   same material, with the memory hierarchy analysis PMPP.4e ch. 5 and 6 reuse.
3. [NVIDIA Deep Learning Performance guides][nv-dl-perf]: how GEMM shapes,
   tile quantization, and wave quantization set Tensor Core efficiency.
4. Golub and Van Loan, _Matrix Computations_: factorizations, least squares,
   and eigenvalue algorithms. Not free.

[how-to-gemm]: https://github.com/flame/how-to-optimize-gemm/wiki
[nv-dl-perf]: https://docs.nvidia.com/deeplearning/performance/index.html
[pfhp]: https://www.cs.utexas.edu/~flame/laff/pfhp/

## Sparse linear algebra and iterative solvers

Supports PMPP.4e ch. 14 and the conjugate gradient loop in ch. 17.

1. [Iterative Methods for Sparse Linear Systems][saad]: free second edition.
   Sparse storage formats, Krylov methods, and preconditioning.
2. [SuiteSparse Matrix Collection][suitesparse]: real sparse matrices for
   testing SpMV kernels on realistic sparsity patterns.
3. Davis, _Direct Methods for Sparse Linear Systems_: CSC storage and sparse
   factorization. SIAM; not free.

[saad]: https://www-users.cse.umn.edu/~saad/IterMethBook_2ndEd.pdf
[suitesparse]: https://sparse.tamu.edu/

## Signal processing: convolution and FFT

Supports PMPP.4e ch. 7 and the Fourier transforms in ch. 17.

1. [But what is a convolution?][3b1b-conv]: discrete convolution, the
   connection to polynomial multiplication, and why the FFT makes it fast.
2. [The Scientist and Engineer's Guide to Digital Signal
   Processing][dspguide]: free book. Convolution, the DFT, the FFT, and
   filters, written for practitioners.

[3b1b-conv]: https://www.3blue1brown.com/lessons/convolutions
[dspguide]: https://www.dspguide.com/

## Partial differential equations and stencils

Supports PMPP.4e ch. 8. A stencil is a finite-difference approximation of a
differential operator applied on a grid.

1. [CFD Python: 12 steps to Navier-Stokes][barba-blog] with the
   [notebooks][cfd-python]: derives finite-difference stencils for diffusion,
   convection, Burgers', Laplace, and Poisson equations and codes them in
   NumPy.
2. LeVeque, _Finite Difference Methods for Ordinary and Partial Differential
   Equations_: accuracy, stability, and the stencils themselves. SIAM; not
   free.

[barba-blog]: https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/
[cfd-python]: https://github.com/barbagroup/CFDPython

## Parallel primitives

Supports PMPP.4e ch. 10 to 13 and 15. Reduction, scan, sort, and graph traversal
are building blocks for the other chapters and for every ML framework.

1. [Prefix Sums and Their Applications][blelloch]: scan as a primitive and its
   uses in sorting, stream compaction, and sparse products.
2. [Parallel Prefix Sum (Scan) with CUDA][gpu-gems-scan]: the work-efficient
   GPU scan that PMPP.4e ch. 11 builds on.
3. [Scalable GPU Graph Traversal][merrill-bfs]: the frontier-based BFS with
   prefix-sum work distribution behind PMPP.4e ch. 15.
4. [CCCL][cccl]: production reduce, scan, sort, histogram, and merge
   implementations in CUB and Thrust. Read them after writing your own.

[blelloch]: https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf
[cccl]: https://nvidia.github.io/cccl/
[gpu-gems-scan]: https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
[merrill-bfs]: https://research.nvidia.com/publication/2012-02_scalable-gpu-graph-traversal

## Application case studies

Supports PMPP.4e ch. 17 and 18. Each is a well-known scientific computing
problem; the PMPP.4e chapters cite the original papers.

- Iterative MRI reconstruction (PMPP.4e ch. 17): non-Cartesian sampling turns
  image reconstruction into a regularized least-squares problem solved with
  conjugate gradient. Background: least squares and Krylov methods in
  [Saad][saad], the DFT in the [DSP guide][dspguide].
- Electrostatic potential maps (PMPP.4e ch. 18): direct Coulomb summation is an
  N-body problem; cutoff summation and binning trade accuracy for locality.
  Stone et al., "Accelerating molecular modeling applications with graphics
  processors", _J. Comput. Chem._ 28 (2007), is the source paper; the publisher
  blocks automated access.
- Monte Carlo methods and random numbers: see cuRAND in
  [`cuda.md`](./cuda.md) and the probability section of
  [`math.md`](./math.md).
