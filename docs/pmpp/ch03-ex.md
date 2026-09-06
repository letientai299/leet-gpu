# Chapter 3 exercises — Multidimensional grids and data

Back to the [reading checklist][reading-checklist].

[reading-checklist]: ./readme.md#3-multidimensional-grids-and-data

Questions transcribed from the book.

## Ex 3.1

In this chapter we implemented a matrix multiplication kernel that has each
thread produce one output matrix element. In this question, you will implement
different matrix-matrix multiplication kernels and compare them.

- a. Write a kernel that has each thread produce one output matrix row. Fill in
  the execution configuration parameters for the design.
- b. Write a kernel that has each thread produce one output matrix column. Fill
  in the execution configuration parameters for the design.
- c. Analyze the pros and cons of each of the two kernel designs.

**Answer:**

See (a) [`matmul.row.cu`][mm-row] and (b) [`matmul.col.cu`][mm-col].

For (c), the analysis below is purely from my understanding after completing
the code, before testing via [nvBench][nvb].

<table>
<tr>
<td></td>
<td>

$$
B = \begin{pmatrix}
b_{11} & b_{12} & \colorbox{lightgray}{$b_{13}$} \\
b_{21} & b_{22} & \colorbox{lightgray}{$b_{23}$}
\end{pmatrix}_{2\times 3}
$$

</td>
</tr>
<tr>
<td>

$$
A = \begin{pmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22} \\
\colorbox{lightgray}{$a_{31}$} & \colorbox{lightgray}{$a_{32}$} \\
a_{41} & a_{42}
\end{pmatrix}_{4\times 2}
$$

</td>
<td>

$$
C = \begin{pmatrix}
c_{11} & c_{12} & c_{13} \\
c_{21} & c_{22} & c_{23} \\
c_{31} & c_{32} & \colorbox{lightgray}{$c_{33}$} \\
c_{41} & c_{42} & c_{43}
\end{pmatrix}_{4\times 3}
$$

</td>
</tr>
</table>

Hardware memory access is fastest with row-major indexing. Hence, accessing
rows of $A$ is fast, but accessing columns of $B$ is slow. In the CPU world, we
have the cache-line invalidation problem. I believe GPU VRAM behaves similarly.
We can't just access a few bytes inside VRAM without locking a larger region.

All 3 solutions need $k \times \mathrm{height}$ random accesses for $B$.

- [1 thread per cell][mm-cell]: needs $\mathrm{width} \times \mathrm{height}$
  random-order accesses to the $C$ data. Even though this solution is the most
  parallel, I guess it would hit the most memory contention. Thus, slowest.
- `matmul.row`: each thread owns a row of $C$, thus reducing contention, and
  because they access a linear region of VRAM, this should be fastest.
- `matmul.col`: each thread owns a column of $C$, and needs to jump between
  cells, so this is slower than `matmul.row`.

### AI verdict

(a)/(b) kernels match the exercise. The inner-product picture (row of $A$ ×
column of $B$) is right, as is row-major: a row of $A$/$C$ is contiguous, a
column of $B$ is strided.

The (c) ranking does not hold. GPU cost is a warp (consecutive threads, same
instruction) touching scattered addresses, not CPU-style cache-line locks, and
not “contention” from many writes to different $C$ cells. Sequential walking by
one thread later is not coalescing across the warp.

- 1 thread / cell: `threadIdx.x` is the column, so neighbors write neighboring
  $C$ and read neighboring $B$. Thread count is $\mathrm{height}\times
  \mathrm{width}$.
- `matmul.row`: $\mathrm{height}$ threads; neighbors own neighbor rows → strided
  $A$/$C$ at one inner-loop step. Each thread covers $\mathrm{width}$ cells.
- `matmul.col`: $\mathrm{width}$ threads; neighbors own neighbor columns →
  adjacent $B$/$C$. Each thread covers $\mathrm{height}$ cells.

“Most parallel ⇒ most contention ⇒ slowest” is backwards. The cell kernel
should win whenever $\mathrm{height}\times\mathrm{width}$ is large enough to
occupy the GPU.

Row vs col is a shape trade: occupancy ($\mathrm{height}$ vs $\mathrm{width}$
threads) against coalescing (col is better for row-major $B$/$C$). Prefer
`matmul.row` when $C$ is tall and skinny ($\mathrm{height}\gg\mathrm{width}$):
more threads can outweigh strided accesses. Prefer `matmul.col` when $C$ is
wide ($\mathrm{width}\gg\mathrm{height}$): more threads *and* coalesced
neighbors. Near-square, col’s access pattern is the cleaner default; measure.

[mm-cell]: ../../src/pmpp/ch03/matmul.cu
[mm-row]: ../../src/pmpp/ch03/matmul.row.cu
[mm-col]: ../../src/pmpp/ch03/matmul.col.cu
[nvb]: https://github.com/NVIDIA/nvbench

## Ex 3.2

A matrix-vector multiplication takes an input matrix $B$ and a vector $C$ and
produces one output vector $A$. Each element of the output vector $A$ is the dot
product of one row of the input matrix $B$ and $C$:

$$
A[i] = \sum_j B[i][j] \, C[j]
$$

For simplicity we will handle only square matrices whose elements are
single-precision floating-point numbers. Write a matrix-vector multiplication
kernel and the host stub function that can be called with four parameters:
pointer to the output vector, pointer to the input matrix, pointer to the input
vector, and the number of elements in each dimension. Use one thread to
calculate an output vector element.

**Answer:** [`matmul.vec.cu`][ex-3-2]

[ex-3-2]: ../../src/pmpp/ch03/matmul.vec.cu

## Ex 3.3

Consider the following CUDA kernel and the corresponding host function that
calls it:

```c:line-numbers {5}
__global__ void foo_kernel(float* a, float* b, unsigned int M, unsigned int N) {
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    b[row * N + col] = a[row * N + col] / 2.1f + 4.8f;
  }
}
void foo(float* a_d, float* b_d) {
  unsigned int M = 150;
  unsigned int N = 300;
  dim3 bd(16, 32);
  dim3 gd((N - 1) / 16 + 1, (M - 1) / 32 + 1);
  foo_kernel<<<gd, bd>>>(a_d, b_d, M, N);
}
```

- a. What is the number of threads per block?
- b. What is the number of threads in the grid?
- c. What is the number of blocks in the grid?
- d. What is the number of threads that execute the code on line 05?

**Answer:**

- a. thread per block $ 16 \times 32 = 512$
- b. threads in grid $ 512 \times 19 \times 5 = 48640 $
- c. blocks in grid $19 \times 5 = 95$
- d. line 5 run count $ 300 \times 150 = 45000$

## Ex 3.4

Consider a 2D matrix with a width of $400$ and a height of $500$. The matrix is
stored as a one-dimensional array. Specify the array index of the matrix element
at row $20$ and column $10$:

- a. If the matrix is stored in row-major order.
- b. If the matrix is stored in column-major order.

**Answer:** Index of $A_{20,10}$ is

- a. row-major: $20 \times 400 + 10 = 8010$
- b. column-major: $10 \times 500 + 20 = 5020$

## Ex 3.5

Consider a 3D tensor with a width of $400$, a height of $500$, and a depth of
$300$. The tensor is stored as a one-dimensional array in row-major order.
Specify the array index of the tensor element at $x = 10$, $y = 20$, and
$z = 5$.

**Answer:** Index of $A_{10, 20, 5}$ is

$$
\begin{array}{rll}
       & 400 \times 500 & \text{Cube layer size}
\\ \times & 5 & \text{Depth}
\\ + & 20 \times 400 + 10 & \text{Index at the element's layer}
\\ \hline
= & 1{,}008{,}010 &
\end{array}
$$
