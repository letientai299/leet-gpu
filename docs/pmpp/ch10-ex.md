# Chapter 10 exercises — Reduction

Back to the [reading checklist][reading-checklist].

[reading-checklist]: ./readme.md#10-reduction

Questions transcribed from the book.

## Ex 10.1

For the simple reduction kernel in Fig. 10.6, if the number of elements is
$1024$ and the warp size is $32$, how many warps in the block will have
divergence during the fifth iteration?

**Answer:**

## Ex 10.2

For the improved reduction kernel in Fig. 10.9, if the number of elements is
$1024$ and the warp size is $32$, how many warps will have divergence during
the fifth iteration?

**Answer:**

## Ex 10.3

Modify the kernel in Fig. 10.9 to use the access pattern illustrated in the
book (the figure immediately below the question).

**Answer:**

## Ex 10.4

Modify the kernel in Fig. 10.15 to perform a max reduction instead of a sum
reduction.

**Answer:**

## Ex 10.5

Modify the kernel in Fig. 10.15 to work for an arbitrary length input that is
not necessarily a multiple of `COARSE_FACTOR*2*blockDim.x`. Add an extra
parameter $N$ to the kernel that represents the length of the input.

**Answer:**

## Ex 10.6

Assume that parallel reduction is to be applied on the following input array:

```math
\boxed{6} \boxed{2} \boxed{7} \boxed{4} \boxed{5} \boxed{8} \boxed{3} \boxed{1}
```

Show how the contents of the array change after each iteration if:

- a. The unoptimized kernel in Fig. 10.6 is used. Initial array:
  $\boxed{6}\boxed{2}\boxed{7}\boxed{4}\boxed{5}\boxed{8}\boxed{3}\boxed{1}$
- b. The kernel optimized for coalescing and divergence in Fig. 10.9 is used.
  Initial array:
  $\boxed{6}\boxed{2}\boxed{7}\boxed{4}\boxed{5}\boxed{8}\boxed{3}\boxed{1}$

**Answer:**
