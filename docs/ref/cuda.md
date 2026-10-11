# CUDA reading list

## Required guides

Read these current NVIDIA guides in order:

1. [CUDA Programming Guide][programming-guide]: programming model, execution,
   memory, synchronization, CUDA C++, and hardware behavior. Treat this as the
   primary source for CUDA semantics.
2. [CUDA C++ Best Practices Guide][best-practices]: correctness, profiling,
   memory access, resource use, and performance optimization.
3. [CUDA Runtime API][runtime-api]: exact API contracts, synchronization rules,
   error handling, types, and function references.
4. [Nsight Compute Profiling Guide][profiling-guide]: kernel metrics,
   bottleneck analysis, replay behavior, and evidence-based optimization.

## Course textbook

- [Programming Massively Parallel Processors: A Hands-on Approach, Fourth
  Edition][pmpp]: concepts, parallel patterns, case studies, and exercises used
  by this repository. Prefer the current NVIDIA guides when the book and the
  installed CUDA version differ.

## Python study path

Study the layers in this order. The later kernel DSLs assume the CUDA execution
and memory model from the required guides.

1. [Introduction to CUDA Python][cuda-python-intro]: map Python tools to the
   CUDA programming model and choose the right abstraction level.
2. [`cuda.core`][cuda-core]: manage devices, memory, streams, events, programs,
   launches, and CUDA graphs from Python.
3. [CuPy][cupy]: use NumPy- and SciPy-style GPU arrays, interoperability, and
   user-defined kernels.
4. [Numba CUDA MLIR][numba-cuda-mlir]: write SIMT kernels in Python. This is the
   evolving successor to Numba-CUDA, which is in maintenance mode.
5. [Triton tutorials][triton]: write tile-oriented ML kernels, starting with
   vector addition, fused softmax, GEMM, and fused attention.
6. [CUTLASS Python and CuTe DSL][cute-dsl]: study layouts, tensors, copy and MMA
   atoms, tiled operations, asynchronous pipelines, and architecture-specific
   Tensor Core kernels.
7. [PyTorch custom C++ and CUDA operators][pytorch-ops]: integrate custom
   kernels with the dispatcher, autograd, fake tensors, and `torch.compile`.

## CUDA engineering practice

### Correctness and debugging

- [Asynchronous execution][async-execution]: understand streams, events,
  synchronization, pinned transfers, allocation lifetimes, and delayed error
  reporting before debugging concurrent code.
- [Compute Sanitizer][compute-sanitizer]: run `memcheck`, `racecheck`,
  `initcheck`, and `synccheck` before trusting kernel results.
- Compare GPU results with a simple CPU or library reference. Test boundary
  sizes, non-multiple tile sizes, aliasing, and supported data types.

### Profiling and performance

- [Nsight Systems][nsight-systems]: locate CPU, transfer, launch,
  synchronization, and communication bottlenecks across the full application.
  Add [NVTX ranges][nsight-systems] around meaningful phases.
- [Nsight Compute][profiling-guide]: analyze one kernel after system profiling
  identifies it as important.
- Measure a stable baseline and effective bandwidth or throughput before
  changing code. Recheck correctness and end-to-end performance afterward.

### Numerical behavior

- [Floating Point and IEEE 754][floating-point]: understand rounding,
  non-associativity, fused multiply-add, accumulation precision, and numerical
  comparisons.
- Treat TF32, FP16, BF16, FP8, and FP4 as accuracy and scaling choices, not only
  performance choices. Check determinism requirements separately.

### Compilation and deployment

- [`nvcc`][nvcc]: understand host and device compilation, `ptxas`, virtual and
  real architectures, fat binaries, relocatable device code, and JIT behavior.
- [CUDA Compatibility][cuda-compatibility]: distinguish the driver, toolkit,
  runtime, libraries, compute capability, and supported compatibility paths.
- Inspect [PTX][ptx] and [SASS][binary-utilities] when compiler decisions,
  register pressure, spills, or instruction selection affect performance.

### Interoperability and scaling

- [CUDA Python interoperability][cuda-interop]: preserve device, context,
  stream, ownership, and lifetime rules across CUDA Array Interface, DLPack,
  CuPy, and PyTorch boundaries.
- [Multi-GPU systems][multi-gpu]: study topology, peer access, CPU affinity,
  process models, communication overlap, [NCCL][nccl], and GPUDirect before
  scaling a workload beyond one GPU.

## ML and LLM CUDA libraries

Study these libraries by role instead of treating all of them as prerequisites.

### Compute and kernel construction

- [cuBLAS and cuBLASLt][cublas]: dense linear algebra and configurable GEMM.
  GEMM is the main compute primitive behind transformer linear layers.
- [cuDNN][cudnn]: optimized attention, convolution, normalization, matmul, and
  graph execution for training and inference.
- [Transformer Engine][transformer-engine]: low-precision transformer layers,
  fused operations, and FP8 or FP4 training and inference.
- [CUTLASS][cutlass]: C++ templates and Python DSLs for custom GEMM, attention,
  data movement, and Tensor Core kernels.

### Distributed execution

- [NCCL][nccl]: topology-aware collectives and point-to-point communication for
  data, tensor, and pipeline parallelism across GPUs and nodes.
- [NVSHMEM][nvshmem]: symmetric memory and GPU-initiated communication. Study
  it after NCCL when kernels need fine-grained or one-sided communication.

### Inference

- [TensorRT][tensorrt]: graph optimization, quantization, engine building, and
  deployment of general deep-learning inference.
- [TensorRT LLM][tensorrt-llm]: LLM-specific inference, batching, KV caches,
  quantization, and multi-GPU or multi-node execution.

### General and workload-specific libraries

- [CCCL][cccl]: `libcu++`, CUB, and Thrust primitives for reductions, scans,
  sorting, synchronization, and reusable custom-kernel building blocks.
- [cuSPARSE][cusparse]: sparse vector and matrix operations, including SpMM.
- [cuFFT][cufft]: FFT operations used by signal, vision, and spectral models.
- [cuRAND][curand]: host- and device-side random-number generation.

## Advanced references

- [PTX ISA][ptx]: virtual instruction-set semantics and compiler output.
- [CUDA Binary Utilities][binary-utilities]: inspect native SASS with
  `cuobjdump` and `nvdisasm`, including architecture-specific instruction
  references. SASS targets physical GPU architectures; PTX is the stable
  virtual ISA.
- [CUDA Toolkit documentation][cuda-docs]: architecture-specific tuning guides,
  library manuals, debugging tools, compatibility guides, and release notes.

[async-execution]: https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html
[best-practices]: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
[binary-utilities]: https://docs.nvidia.com/cuda/cuda-binary-utilities/
[cccl]: https://nvidia.github.io/cccl/
[cublas]: https://docs.nvidia.com/cuda/cublas/
[compute-sanitizer]: https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html
[cuda-compatibility]: https://docs.nvidia.com/deploy/cuda-compatibility/latest/
[cuda-docs]: https://docs.nvidia.com/cuda/
[cuda-core]: https://nvidia.github.io/cuda-python/cuda-core/latest/
[cuda-interop]: https://nvidia.github.io/cuda-python/cuda-core/latest/interoperability.html
[cuda-python-intro]: https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-python.html
[cudnn]: https://docs.nvidia.com/deeplearning/cudnn/latest/
[cufft]: https://docs.nvidia.com/cuda/cufft/
[cupy]: https://docs.cupy.dev/en/stable/
[curand]: https://docs.nvidia.com/cuda/curand/
[cusparse]: https://docs.nvidia.com/cuda/cusparse/
[cute-dsl]: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html
[cutlass]: https://docs.nvidia.com/cutlass/latest/
[floating-point]: https://docs.nvidia.com/cuda/floating-point/
[multi-gpu]: https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html
[nccl]: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/
[nsight-systems]: https://docs.nvidia.com/nsight-systems/UserGuide/
[numba-cuda-mlir]: https://nvidia.github.io/numba-cuda-mlir/latest/
[nvcc]: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/
[nvshmem]: https://docs.nvidia.com/nvshmem/api/latest/introduction.html
[pmpp]: https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0
[profiling-guide]: https://docs.nvidia.com/nsight-compute/ProfilingGuide/
[programming-guide]: https://docs.nvidia.com/cuda/cuda-programming-guide/
[ptx]: https://docs.nvidia.com/cuda/parallel-thread-execution/
[pytorch-ops]: https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
[runtime-api]: https://docs.nvidia.com/cuda/cuda-runtime-api/
[tensorrt]: https://docs.nvidia.com/deeplearning/tensorrt/latest/
[tensorrt-llm]: https://nvidia.github.io/TensorRT-LLM/
[transformer-engine]: https://docs.nvidia.com/deeplearning/transformer-engine/
[triton]: https://triton-lang.org/main/getting-started/tutorials/
