# Reference reading lists

One file per topic. Open the file that matches the question, then follow its
links. Every file lists free or official sources first.

## Source tags

A chapter reference names its source with a short tag, as in `PMPP.4e ch. 7`.
Tags let one topic file point at several books or courses studied in this
repository. Add a tag here when a new source joins.

- `PMPP.4e`: _Programming Massively Parallel Processors_, fourth edition
  (Hwu, Kirk, El Hajj). Notes in [`../pmpp/`](../pmpp/readme.md).

## Files

- [`cuda.md`](./cuda.md): NVIDIA guides, Python kernel DSLs, profiling, numerics
  on GPUs, and the CUDA libraries behind ML and LLM stacks.
- [`math.md`](./math.md): linear algebra, calculus, probability, information
  theory, optimization, and algorithms. Start here when a source chapter uses a
  concept you have not met.
- [`numerical.md`](./numerical.md): floating point, dense and sparse linear
  algebra, signal processing, PDE stencils, parallel primitives, and the
  scientific applications in the PMPP.4e case studies.
- [`ml.md`](./ml.md): classical machine learning and statistical learning.
- [`deep-learning.md`](./deep-learning.md): neural networks, backpropagation,
  convolutional networks, training, and the performance guides that connect
  layers to GEMM and convolution kernels.
- [`llm.md`](./llm.md): transformers, attention kernels, training at scale,
  inference, and quantization.

## Routing by question

- How does CUDA, a library, or a tool work: `cuda.md`.
- What is the math behind an operation (matrix product, gradient, expectation,
  convolution theorem): `math.md`.
- Why does a result differ from the reference, or how to pick a precision:
  `numerical.md`, then the numerical behavior section of `cuda.md`.
- How a PMPP.4e pattern (reduction, scan, sort, SpMV, BFS, stencil) is defined
  and where it is used: `numerical.md`.
- What a model does with the kernel (loss, layer, optimizer): `ml.md` for
  classical models, `deep-learning.md` for networks, `llm.md` for transformers.
- How kernels compose into training or serving systems: `llm.md`.

## PMPP.4e chapters to topics

Each item names the PMPP.4e computation, the field it comes from, and the file
that covers the background. Chapter numbers below refer to PMPP.4e.

- Matrix multiplication and tiling (ch. 3 to 6): linear algebra in `math.md`;
  blocked GEMM and roofline reasoning in `numerical.md`; GEMM as the transformer
  primitive in `deep-learning.md` and `llm.md`.
- Convolution (ch. 7): signal processing and the convolution theorem in
  `numerical.md`; convolutional layers and their GEMM lowering in
  `deep-learning.md`.
- Stencil (ch. 8): finite-difference PDE discretization in `numerical.md`.
- Histogram (ch. 9): probability and empirical distributions in `math.md`;
  quantization calibration in `llm.md`.
- Reduction and prefix sum (ch. 10 and 11): parallel primitives in
  `numerical.md`; softmax, normalization, and online reductions in `llm.md`.
- Merge and sort (ch. 12 and 13): algorithms in `math.md`; parallel primitives
  in `numerical.md`.
- Sparse matrix computation (ch. 14): sparse formats and iterative solvers in
  `numerical.md`; sparsity in models in `llm.md`.
- Graph traversal (ch. 15): graph algorithms in `math.md`; GPU BFS in
  `numerical.md`.
- Deep learning (ch. 16): `deep-learning.md`.
- Iterative MRI reconstruction (ch. 17): least squares, conjugate gradient, and
  Fourier transforms in `numerical.md`.
- Electrostatic potential map (ch. 18): N-body and cutoff summation in
  `numerical.md`.
- Heterogeneous clusters (ch. 20): distributed execution in `cuda.md`; data,
  tensor, and pipeline parallelism in `llm.md`.
- Numerical considerations (appendix A): floating point in `numerical.md` and
  `cuda.md`.
