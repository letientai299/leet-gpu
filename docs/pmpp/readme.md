# Programming Massively Parallel Processors (4th Edition)

Hwu, Kirk & El Hajj (2022). Reading-progress checklist.

## Part I — Fundamental Concepts

### 1. Introduction

- [x] 1.1 Heterogeneous parallel computing
  - GPU apps is expected to be written with a large number of parallel threads.
  - Throughput oriented design.
- [x] 1.2 Why more speed or parallelism?
  - The book aim to:
    - Teach data management techniques
    - Provide many practical code examples and exercises.
- [x] 1.3 Speeding up real applications
  - Amdahl's Law: speedup via parallelism is limited by the portion of
    parallelizable part.
- [x] 1.4 Challenges in parallel programming
  - Design parallel algo
  - Memory bound applications
  - Input data characteristic.
  - Synchronization overhead.
- [x] 1.5 Related parallel programming interfaces
  - OpenMP
  - MPI
  - OpenCL
- [x] 1.6 Overarching goals
  - Understand hardware architecture and computational thinking
  - Correctness, reliability, debugability
- [x] 1.7 Organization of the book

### 2. Heterogeneous data parallel computing

- [x] 2.1 Data parallelism
  - Data parallelism
  - Task parallelism
- [x] 2.2 CUDA C program structure
- [x] 2.3 A vector addition kernel
  - [`../../apps/pmpp/ch02/vec-add.cu`](../../apps/pmpp/ch02/vec-add.cu)
- [x] 2.4 Device global memory and data transfer
- [x] 2.5 Kernel functions and threading
  - 2d or 3d thread blocks be picked by data shape.
  - Use multiple of 32 for each dimention for hardware efficiency
  - `__global__`: call from CPU, run on GPU
  - `__device__`: call from GPU, run on GPU
  - `__host__`: default, call and run on CPU, exists to support tagging a
    function together `__device__`, make it call or run on both hardwares
  - There's `cudaMallocManaged` for Unified Memory access, auto migrate data, no
    need flags like `cudaMemcpyDeviceToHost`, trade explicit control effort for
    runtime overhead.
- [x] 2.6 Calling kernel functions
- [x] 2.7 Compilation
- [x] 2.8 Summary
- [x] [Exercises](./ch02-ex.md)

### 3. Multidimensional grids and data

- [x] 3.1 Multidimensional grid organization
  - Until now,this book discuss only grid and block of thread. In CPG (CUDA
    programming guide, v13.3), we know that that `grid > cluster > block`.
- [x] 3.2 Mapping threads to multidimensional data
  - [`../../apps/pmpp/ch03/grayscale.cu`](../../apps/pmpp/ch03/grayscale.cu)
- [ ] 3.3 Image blur: a more complex kernel
  - _Convolution pattern_: In math, an operation on 2 funcs $f$ and $g$, produce
    3rd func $f*g$. In this chapter, it's about a square sliding filter on the
    image.
- [ ] 3.4 Matrix multiplication
- [ ] 3.5 Summary
- [ ] [Exercises](./ch03-ex.md)

### 4. Compute architecture and scheduling

- [ ] 4.1 Architecture of a modern GPU
- [ ] 4.2 Block scheduling
- [ ] 4.3 Synchronization and transparent scalability
- [ ] 4.4 Warps and SIMD hardware
- [ ] 4.5 Control divergence
- [ ] 4.6 Warp scheduling and latency tolerance
- [ ] 4.7 Resource partitioning and occupancy
- [ ] 4.8 Querying device properties
- [ ] 4.9 Summary
- [ ] [Exercises](./ch04-ex.md)

### 5. Memory architecture and data locality

- [ ] 5.1 Importance of memory access efficiency
- [ ] 5.2 CUDA memory types
- [ ] 5.3 Tiling for reduced memory traffic
- [ ] 5.4 A tiled matrix multiplication kernel
- [ ] 5.5 Boundary checks
- [ ] 5.6 Impact of memory usage on occupancy
- [ ] 5.7 Summary
- [ ] [Exercises](./ch05-ex.md)

### 6. Performance considerations

- [ ] 6.1 Memory coalescing
- [ ] 6.2 Hiding memory latency
- [ ] 6.3 Thread coarsening
- [ ] 6.4 A checklist of optimizations
- [ ] 6.5 Knowing your computation's bottleneck
- [ ] 6.6 Summary
- [ ] [Exercises](./ch06-ex.md)

## Part II — Parallel Patterns

### 7. Convolution

- [ ] 7.1 Background
- [ ] 7.2 Parallel convolution: a basic algorithm
- [ ] 7.3 Constant memory and caching
- [ ] 7.4 Tiled convolution with halo cells
- [ ] 7.5 Tiled convolution using caches for halo cells
- [ ] 7.6 Summary
- [ ] [Exercises](./ch07-ex.md)

### 8. Stencil

- [ ] 8.1 Background
- [ ] 8.2 Parallel stencil: a basic algorithm
- [ ] 8.3 Shared memory tiling for stencil sweep
- [ ] 8.4 Thread coarsening
- [ ] 8.5 Register tiling
- [ ] 8.6 Summary
- [ ] [Exercises](./ch08-ex.md)

### 9. Parallel histogram

- [ ] 9.1 Background
- [ ] 9.2 Atomic operations and a basic histogram kernel
- [ ] 9.3 Latency and throughput of atomic operations
- [ ] 9.4 Privatization
- [ ] 9.5 Coarsening
- [ ] 9.6 Aggregation
- [ ] 9.7 Summary
- [ ] [Exercises](./ch09-ex.md)

### 10. Reduction

- [ ] 10.1 Background
- [ ] 10.2 Reduction trees
- [ ] 10.3 A simple reduction kernel
- [ ] 10.4 Minimizing control divergence
- [ ] 10.5 Minimizing memory divergence
- [ ] 10.6 Minimizing global memory accesses
- [ ] 10.7 Hierarchical reduction for arbitrary input length
- [ ] 10.8 Thread coarsening for reduced overhead
- [ ] 10.9 Summary
- [ ] [Exercises](./ch10-ex.md)

### 11. Prefix sum (scan)

- [ ] 11.1 Background
- [ ] 11.2 Parallel scan with the Kogge-Stone algorithm
- [ ] 11.3 Speed and work efficiency consideration
- [ ] 11.4 Parallel scan with the Brent-Kung algorithm
- [ ] 11.5 Coarsening for even more work efficiency
- [ ] 11.6 Segmented parallel scan for arbitrary-length inputs
- [ ] 11.7 Single-pass scan for memory access efficiency
- [ ] 11.8 Summary
- [ ] [Exercises](./ch11-ex.md)

### 12. Merge

- [ ] 12.1 Background
- [ ] 12.2 A sequential merge algorithm
- [ ] 12.3 A parallelization approach
- [ ] 12.4 Co-rank function implementation
- [ ] 12.5 A basic parallel merge kernel
- [ ] 12.6 A tiled merge kernel to improve coalescing
- [ ] 12.7 A circular buffer merge kernel
- [ ] 12.8 Thread coarsening for merge
- [ ] 12.9 Summary
- [ ] [Exercises](./ch12-ex.md)

## Part III — Advanced Patterns and Applications

### 13. Sorting

- [ ] 13.1 Background
- [ ] 13.2 Radix sort
- [ ] 13.3 Parallel radix sort
- [ ] 13.4 Optimizing for memory coalescing
- [ ] 13.5 Choice of radix value
- [ ] 13.6 Thread coarsening to improve coalescing
- [ ] 13.7 Parallel merge sort
- [ ] 13.8 Other parallel sort methods
- [ ] 13.9 Summary
- [ ] [Exercises](./ch13-ex.md)

### 14. Sparse matrix computation

- [ ] 14.1 Background
- [ ] 14.2 A simple SpMV kernel with the COO format
- [ ] 14.3 Grouping row nonzeros with the CSR format
- [ ] 14.4 Improving memory coalescing with the ELL format
- [ ] 14.5 Regulating padding with the hybrid ELL-COO format
- [ ] 14.6 Reducing control divergence with the JDS format
- [ ] 14.7 Summary
- [ ] [Exercises](./ch14-ex.md)

### 15. Graph traversal

- [ ] 15.1 Background
- [ ] 15.2 Breadth-first search
- [ ] 15.3 Vertex-centric parallelization of breadth-first search
- [ ] 15.4 Edge-centric parallelization of breadth-first search
- [ ] 15.5 Improving efficiency with frontiers
- [ ] 15.6 Reducing contention with privatization
- [ ] 15.7 Other optimizations
- [ ] 15.8 Summary
- [ ] [Exercises](./ch15-ex.md)

### 16. Deep learning

- [ ] 16.1 Background
- [ ] 16.2 Convolutional neural networks
- [ ] 16.3 Convolutional layer: a CUDA inference kernel
- [ ] 16.4 Formulating a convolutional layer as GEMM
- [ ] 16.5 CUDNN library
- [ ] 16.6 Summary
- [ ] [Exercises](./ch16-ex.md)

### 17. Iterative magnetic resonance imaging reconstruction

- [ ] 17.1 Background
- [ ] 17.2 Iterative reconstruction
- [ ] 17.3 Computing FHD
- [ ] 17.4 Summary
- [ ] [Exercises](./ch17-ex.md)

### 18. Electrostatic potential map

- [ ] 18.1 Background
- [ ] 18.2 Scatter versus gather in kernel design
- [ ] 18.3 Thread coarsening
- [ ] 18.4 Memory coalescing
- [ ] 18.5 Cutoff binning for data size scalability
- [ ] 18.6 Summary
- [ ] [Exercises](./ch18-ex.md)

### 19. Parallel programming and computational thinking

- [ ] 19.1 Goals of parallel computing
- [ ] 19.2 Algorithm selection
- [ ] 19.3 Problem decomposition
- [ ] 19.4 Computational thinking
- [ ] 19.5 Summary
- [ ] [Exercises](./ch19-ex.md)

## Part IV — Advanced Practices

### 20. Programming a heterogeneous computing cluster

- [ ] 20.1 Background
- [ ] 20.2 A running example
- [ ] 20.3 Message passing interface basics
- [ ] 20.4 Message passing interface point-to-point communication
- [ ] 20.5 Overlapping computation and communication
- [ ] 20.6 Message passing interface collective communication
- [ ] 20.7 CUDA aware message passing interface
- [ ] 20.8 Summary
- [ ] [Exercises](./ch20-ex.md)

### 21. CUDA dynamic parallelism

- [ ] 21.1 Background
- [ ] 21.2 Dynamic parallelism overview
- [ ] 21.3 An example: Bezier curves
- [ ] 21.4 A recursive example: quadtrees
- [ ] 21.5 Important considerations
- [ ] 21.6 Summary
- [ ] [Exercises](./ch21-ex.md)

### 22. Advanced practices and future evolution

- [ ] 22.1 Model of host/device interaction
- [ ] 22.2 Kernel execution control
- [ ] 22.3 Memory bandwidth and compute throughput
- [ ] 22.4 Programming environment
- [ ] 22.5 Future outlook

### 23. Conclusion and outlook

- [ ] 23.1 Goals revisited
- [ ] 23.2 Future outlook

## Appendix

### A. An introduction to numerical considerations

- [ ] A.1 Floating-point data representation
- [ ] A.2 Representable numbers
- [ ] A.3 Special bit patterns and precision in IEEE format
- [ ] A.4 Arithmetic accuracy and rounding
- [ ] A.5 Algorithm considerations
- [ ] A.6 Linear solvers and numerical stability
- [ ] A.7 Summary
