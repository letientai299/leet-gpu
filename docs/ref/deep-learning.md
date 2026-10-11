# Deep learning reading list

Neural networks, backpropagation, convolutional and recurrent networks,
normalization, optimizers, and mixed-precision training. Supports PMPP.4e ch. 16
and explains what the GEMM, convolution, reduction, and softmax kernels compute.
Transformers continue in [`llm.md`](./llm.md).

## Start

1. [3Blue1Brown Neural Networks][3b1b-nn]: what a network computes and what
   gradient descent and backpropagation do.
2. [Neural Networks: Zero to Hero][zero-to-hero]: Karpathy builds autograd,
   an MLP, and a GPT from scratch in Python. The best single sequence for
   understanding what frameworks do.
3. [CS231n notes][cs231n]: linear classifiers, backpropagation, convolutional
   networks, and training recipes, with the derivations.
4. [Calculus on Computational Graphs: Backpropagation][colah-backprop]: the
   forward-mode and reverse-mode view of the chain rule.

[3b1b-nn]: https://www.3blue1brown.com/?topic=neural-networks
[colah-backprop]: https://colah.github.io/posts/2015-08-Backprop/
[cs231n]: https://cs231n.github.io/
[zero-to-hero]: https://karpathy.ai/zero-to-hero.html

## Books

1. [Dive into Deep Learning][d2l]: free, runnable in PyTorch. Each concept
   comes with code and exercises.
2. [Understanding Deep Learning][udl]: free PDF. Current coverage, including
   transformers, diffusion, and graph networks, with clear figures.
3. [Deep Learning: Foundations and Concepts][bishop-dl]: free online. The
   probabilistic treatment.
4. [Deep Learning][goodfellow]: free online. The 2016 reference; part I is a
   compact math review.

[bishop-dl]: https://www.bishopbook.com/
[d2l]: https://d2l.ai/
[goodfellow]: https://www.deeplearningbook.org/
[udl]: https://udlbook.github.io/udlbook/

## Courses

- [fast.ai Practical Deep Learning][fastai]: top-down. Train models first,
  then open them up.
- [PyTorch tutorials][pytorch-tutorials]: autograd, modules, data loading,
  distributed training, and custom operators.

[fastai]: https://course.fast.ai/
[pytorch-tutorials]: https://docs.pytorch.org/tutorials/

## Papers behind the kernels

Read these after the courses. Each one defines an operation that a CUDA kernel
in this repository or in cuDNN implements.

- [Deep Residual Learning][resnet]: skip connections; why networks got deep.
- [Batch Normalization][batchnorm]: per-channel mean and variance reductions
  inside the network.
- [Adam][adam]: the optimizer update, an elementwise kernel over every
  parameter.
- [cuDNN: Efficient Primitives for Deep Learning][cudnn-paper]: implicit GEMM
  convolution, the lowering PMPP.4e ch. 16 describes.
- [Fast Algorithms for Convolutional Neural Networks][winograd]: Winograd
  minimal filtering, the other fast convolution route besides FFT.
- [Mixed Precision Training][mixed-precision] and NVIDIA's [Mixed Precision
  Training guide][nv-mixed-precision]: FP16 storage, FP32 accumulation, loss
  scaling, and why Tensor Cores need it.

[adam]: https://arxiv.org/abs/1412.6980
[batchnorm]: https://arxiv.org/abs/1502.03167
[cudnn-paper]: https://arxiv.org/abs/1410.0759
[mixed-precision]: https://arxiv.org/abs/1710.03740
[nv-mixed-precision]: https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html
[resnet]: https://arxiv.org/abs/1512.03385
[winograd]: https://arxiv.org/abs/1509.09308

## Performance

- [NVIDIA Deep Learning Performance guides][nv-dl-perf]: GEMM, convolution,
  recurrent, memory-bound, and fully connected layer guides. Each explains
  which dimensions must be multiples of 8 or 16 for Tensor Cores and how to
  reason about arithmetic intensity per layer.
- [GPU MODE lectures][gpu-mode]: recorded talks on kernels for deep learning,
  from profiling PyTorch to writing attention kernels.

[gpu-mode]: https://github.com/gpu-mode/lectures
[nv-dl-perf]: https://docs.nvidia.com/deeplearning/performance/index.html
