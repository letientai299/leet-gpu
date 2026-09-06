# Chapter 9 exercises — Parallel histogram

Back to the [reading checklist][reading-checklist].

[reading-checklist]: ./readme.md#9-parallel-histogram

Questions transcribed from the book.

## Ex 9.1

Assume that each atomic operation in a DRAM system has a total latency of
$100\ \mathrm{ns}$. What is the maximum throughput that we can get for atomic
operations on the same global memory variable?

**Answer:**

## Ex 9.2

For a processor that supports atomic operations in L2 cache, assume that each
atomic operation takes $4\ \mathrm{ns}$ to complete in L2 cache and
$100\ \mathrm{ns}$ to complete in DRAM. Assume that $90$% of the atomic
operations hit in L2 cache. What is the approximate throughput for atomic
operations on the same global memory variable?

**Answer:**

## Ex 9.3

In [Ex 9.1][ex-9-1], assume that a kernel performs five floating-point
operations per atomic operation. What is the maximum floating-point throughput
of the kernel execution as limited by the throughput of the atomic operations?

**Answer:**

## Ex 9.4

In [Ex 9.1][ex-9-1], assume that we privatize the global memory variable into
shared memory variables in the kernel and that the shared memory access latency
is $1\ \mathrm{ns}$. All original global memory atomic operations are converted
into shared memory atomic operation. For simplicity, assume that the additional
global memory atomic operations for accumulating privatized variable into the
global variable adds $10$% to the total execution time. Assume that a kernel
performs five floating-point operations per atomic operation. What is the
maximum floating-point throughput of the kernel execution as limited by the
throughput of the atomic operations?

**Answer:**

## Ex 9.5

To perform an atomic add operation to add the value of an integer variable
`Partial` to a global memory integer variable `Total`, which one of the
following statements should be used?

- a. `atomicAdd(Total, 1);`
- b. `atomicAdd(&Total, &Partial);`
- c. `atomicAdd(Total, &Partial);`
- d. `atomicAdd(&Total, Partial);`

**Answer:**

## Ex 9.6

Consider a histogram kernel that processes an input with $524{,}288$ elements
to produce a histogram with $128$ bins. The kernel is configured with $1024$
threads per block.

- a. What is the total number of atomic operations that are performed on global
  memory by the kernel in Fig. 9.6 where no privatization, shared memory, and
  thread coarsening are used?
- b. What is the maximum number of atomic operations that may be performed on
  global memory by the kernel in Fig. 9.10 where privatization and shared
  memory are used but not thread coarsening?
- c. What is the maximum number of atomic operations that may be performed on
  global memory by the kernel in Fig. 9.14 where privatization, shared memory,
  and thread coarsening are used with a coarsening factor of $4$?

**Answer:**

[ex-9-1]: ./ch09-ex.md#ex-91
