# Math reading list

Study in the order below. Linear algebra and matrix calculus carry most of
PMPP.4e, deep learning, and LLM work. Probability and optimization explain why
models train. Algorithms explain the PMPP.4e sort, merge, and graph chapters.

## Linear algebra

Covers vectors, matrices, matrix multiplication, rank, projections,
eigenvalues, and the SVD. Every GEMM, attention score, and embedding lookup is
a matrix product.

1. [Essence of Linear Algebra][3b1b-la]: short videos that build geometric
   intuition. Watch before reading a textbook.
2. [MIT 18.06 Linear Algebra][mit-1806]: Strang's lectures and problem sets.
   The standard first course.
3. [Immersive Linear Algebra][immersive-la]: interactive figures for the same
   material.
4. [Linear Algebra Done Right][axler]: proof-based treatment. Read after 18.06
   when definitions in papers feel loose.
5. [Advanced Linear Algebra: Foundations to Frontiers][alaff]: numerical linear
   algebra from the group behind BLIS. Bridges math and GEMM implementation.
6. [MIT 18.065 Matrix Methods][mit-18065]: Strang's course on the matrix
   methods used in data analysis and machine learning: SVD, PCA, least squares,
   and gradient descent.
7. [The Matrix Cookbook][matrix-cookbook]: identities, derivatives, and
   decompositions. Keep it open while reading papers.

[3b1b-la]: https://www.3blue1brown.com/?topic=linear-algebra
[alaff]: https://www.cs.utexas.edu/~flame/laff/alaff/ALAFF.html
[axler]: https://linear.axler.net/
[immersive-la]: https://immersivemath.com/ila/index.html
[matrix-cookbook]: https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf
[mit-1806]: https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
[mit-18065]: https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/

## Calculus and matrix calculus

Covers derivatives, gradients, the chain rule, Jacobians, and Taylor
expansion. Backpropagation is the chain rule applied to matrix functions.

1. [Essence of Calculus][3b1b-calc]: intuition for derivatives and integrals.
2. [MIT 18.01 Single Variable Calculus][mit-1801] and [MIT 18.02 Multivariable
   Calculus][mit-1802]: full courses with gradients, partial derivatives, and
   multiple integrals.
3. [The Matrix Calculus You Need For Deep Learning][matrix-calc]: Jacobians,
   vector chain rule, and the gradient of a loss through a layer.

[3b1b-calc]: https://www.3blue1brown.com/?topic=calculus
[matrix-calc]: https://explained.ai/matrix-calculus/
[mit-1801]: https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/
[mit-1802]: https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/

## Probability and statistics

Covers random variables, distributions, expectation, variance, Bayes' rule,
maximum likelihood, and sampling. Loss functions are negative log-likelihoods.
Histograms estimate distributions. Random initialization, dropout, and
sampling tokens all draw from distributions.

1. [Seeing Theory][seeing-theory]: visual introduction to probability and
   inference.
2. [Introduction to Probability, Statistics, and Random
   Processes][prob-course]: free textbook with exercises and solutions.
3. Blitzstein and Hwang, _Introduction to Probability_, with the Harvard Stat
   110 lectures. The standard rigorous first course; the site blocks automated
   access, so search for "Stat 110".
4. [Mathematics for Machine Learning][mml]: one book that covers linear
   algebra, analytic geometry, matrix decompositions, vector calculus,
   probability, and optimization at the level machine learning needs. Read it
   as the consolidation pass after the courses above.

[mml]: https://mml-book.github.io/
[prob-course]: https://www.probabilitycourse.com/
[seeing-theory]: https://seeing-theory.brown.edu/

## Information theory

Covers entropy, cross-entropy, KL divergence, and coding. Cross-entropy is the
LLM training loss; perplexity is its exponential; KL divergence appears in
distillation and alignment objectives.

- [Information Theory, Inference, and Learning Algorithms][mackay]: free book
  that connects coding, probability, and neural networks.

[mackay]: https://www.inference.org.uk/mackay/itila/

## Optimization

Covers convexity, gradient descent, stochastic gradients, line search, and
duality. Training is optimization; quantization and pruning are constrained
optimization.

- [Convex Optimization][boyd]: free book and lecture slides. Chapters 1 to 5
  and 9 cover what deep learning uses.
- Nocedal and Wright, _Numerical Optimization_: nonconvex methods, quasi-Newton
  updates, and trust regions. Not free; read via a library.

[boyd]: https://web.stanford.edu/~boyd/cvxbook/

## Algorithms and discrete math

Covers asymptotic analysis, recursion, sorting, merging, dynamic programming,
and graph search. The PMPP.4e merge, sort, and BFS chapters parallelize these
sequential algorithms.

- Jeff Erickson, _Algorithms_: free book with chapters on recursion, dynamic
  programming, and graph algorithms. The site was unreachable at the time of
  writing; search for "Jeff Erickson Algorithms".
- Cormen, Leiserson, Rivest, and Stein, _Introduction to Algorithms_: the
  reference for sorting, order statistics, and graph algorithms. Not free.
- [Prefix Sums and Their Applications][blelloch]: the scan primitive and the
  problems it solves: radix sort, sparse matrix products, and lexical analysis.

[blelloch]: https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf
