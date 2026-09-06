# Chapter 7 exercises — Convolution

Back to the [reading checklist][reading-checklist].

[reading-checklist]: ./readme.md#7-convolution

Questions transcribed from the book.

## Ex 7.1

Calculate the $P[0]$ value in Fig. 7.3.

**Answer:**

## Ex 7.2

Consider performing a 1D convolution on array
$N = \lbrace 4, 1, 3, 2, 3 \rbrace$ with filter $F = \lbrace 2, 1, 4 \rbrace$.
What is the resulting output array?

**Answer:**

## Ex 7.3

What do you think the following 1D convolution filters are doing?

- a. $[0\ 1\ 0]$
- b. $[0\ 0\ 1]$
- c. $[1\ 0\ 0]$
- d. $[-\frac{1}{2}\ 0\ \frac{1}{2}]$
- e. $[\frac{1}{3}\ \frac{1}{3}\ \frac{1}{3}]$

**Answer:**

## Ex 7.4

Consider performing a 1D convolution on an array of size $N$ with a filter of
size $M$:

- a. How many ghost cells are there in total?
- b. How many multiplications are performed if ghost cells are treated as
  multiplications (by $0$)?
- c. How many multiplications are performed if ghost cells are not treated as
  multiplications?

**Answer:**

## Ex 7.5

Consider performing a 2D convolution on a square matrix of size $N \times N$
with a square filter of size $M \times M$:

- a. How many ghost cells are there in total?
- b. How many multiplications are performed if ghost cells are treated as
  multiplications (by $0$)?
- c. How many multiplications are performed if ghost cells are not treated as
  multiplications?

**Answer:**

## Ex 7.6

Consider performing a 2D convolution on a rectangular matrix of size
$N_1 \times N_2$ with a rectangular mask of size $M_1 \times M_2$:

- a. How many ghost cells are there in total?
- b. How many multiplications are performed if ghost cells are treated as
  multiplications (by $0$)?
- c. How many multiplications are performed if ghost cells are not treated as
  multiplications?

**Answer:**

## Ex 7.7

Consider performing a 2D tiled convolution with the kernel shown in Fig. 7.12
on an array of size $N \times N$ with a filter of size $M \times M$ using an
output tile of size $T \times T$.

- a. How many thread blocks are needed?
- b. How many threads are needed per block?
- c. How much shared memory is needed per block?
- d. Repeat the same questions if you were using the kernel in Fig. 7.15.

**Answer:**

## Ex 7.8

Revise the 2D kernel in Fig. 7.7 to perform 3D convolution.

**Answer:**

## Ex 7.9

Revise the 2D kernel in Fig. 7.9 to perform 3D convolution.

**Answer:**

## Ex 7.10

Revise the tiled 2D kernel in Fig. 7.12 to perform 3D convolution.

**Answer:**
