# Machine learning reading list

Classical machine learning: regression, classification, clustering,
dimensionality reduction, and the statistical ideas behind training and
evaluation. Deep learning and transformers continue in
[`deep-learning.md`](./deep-learning.md) and [`llm.md`](./llm.md).

## Courses

1. [Google Machine Learning Crash Course][google-mlcc]: short lessons on loss,
   gradient descent, generalization, and embeddings. A one-week orientation.
2. [Stanford CS229][cs229] and its [lecture notes][cs229-notes]: supervised
   learning, generative models, kernels, learning theory, and unsupervised
   learning with the math written out.

[cs229]: https://cs229.stanford.edu/
[cs229-notes]: https://cs229.stanford.edu/main_notes.pdf
[google-mlcc]: https://developers.google.com/machine-learning/crash-course

## Books

Read in order of difficulty.

1. [An Introduction to Statistical Learning][islp]: free, with Python and R
   labs. Linear models, trees, SVMs, clustering, and resampling.
2. [The Elements of Statistical Learning][esl]: free. The same topics with the
   theory behind them.
3. [Pattern Recognition and Machine Learning][prml]: free PDF. Bayesian
   methods, graphical models, EM, and variational inference.
4. [Probabilistic Machine Learning: An Introduction][pml1]: free draft.
   Current notation and coverage, including deep learning.

[esl]: https://hastie.su.domains/ElemStatLearn/
[islp]: https://www.statlearning.com/
[pml1]: https://probml.github.io/pml-book/book1.html
[prml]: https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/

## Practice

- [scikit-learn user guide][sklearn]: reference implementations and
  explanations for every classical algorithm. Use it as the CPU oracle when
  checking a GPU kernel for k-means, PCA, or logistic regression.

[sklearn]: https://scikit-learn.org/stable/user_guide.html

## Where the kernels appear

- Linear and logistic regression: GEMM, reductions, and gradient descent.
- k-means and k-nearest neighbors: distance matrices (GEMM), argmin
  reductions, and histograms.
- PCA and SVD: dense eigenvalue and factorization routines; see cuSOLVER in
  the CUDA toolkit documentation linked from [`cuda.md`](./cuda.md).
- Decision trees and gradient boosting: histogram building and prefix sums
  over sorted features.
