# LLM reading list

Transformers, the kernels inside them, training at scale, inference, and
quantization. Assumes the background in
[`deep-learning.md`](./deep-learning.md).

## Transformers

1. [The Illustrated Transformer][illustrated]: the architecture in pictures.
2. [The Annotated Transformer][annotated]: the original paper with a line by
   line PyTorch implementation.
3. [Attention Is All You Need][attention]: the paper. Read after the two
   walkthroughs.
4. [Neural Machine Translation by Jointly Learning to Align and
   Translate][bahdanau]: where attention came from.
5. [BERT][bert] and [Language Models are Few-Shot Learners][gpt3]:
   encoder-only and decoder-only training objectives, and scale as a method.

[annotated]: https://nlp.seas.harvard.edu/annotated-transformer/
[attention]: https://arxiv.org/abs/1706.03762
[bahdanau]: https://arxiv.org/abs/1409.0473
[bert]: https://arxiv.org/abs/1810.04805
[gpt3]: https://arxiv.org/abs/2005.14165
[illustrated]: https://jalammar.github.io/illustrated-transformer/

## Courses and books

1. [Neural Networks: Zero to Hero][zero-to-hero] and [nanoGPT][nanogpt]:
   build GPT-2 from tokenizer to training loop.
2. [Build a Large Language Model (From Scratch)][llms-from-scratch]: Raschka's
   book code. Attention, GPT architecture, pretraining, and fine-tuning in
   plain PyTorch.
3. [Stanford CS336: Language Modeling from Scratch][cs336]: assignments
   implement tokenizers, transformer blocks, FlashAttention in Triton,
   distributed training, and scaling-law fits.
4. [Stanford CS224n][cs224n]: NLP with deep learning; the lecture notes cover
   word vectors through transformers.
5. [Speech and Language Processing][slp3]: free draft of the NLP textbook;
   chapters on n-grams, embeddings, transformers, and LLMs.
6. [Hugging Face LLM Course][hf-llm-course]: the practical layer: tokenizers,
   transformers library, fine-tuning, and deployment.

[cs224n]: https://web.stanford.edu/class/cs224n/
[cs336]: https://cs336.stanford.edu/spring2025/
[hf-llm-course]: https://huggingface.co/learn/llm-course
[llms-from-scratch]: https://github.com/rasbt/LLMs-from-scratch
[nanogpt]: https://github.com/karpathy/nanoGPT
[slp3]: https://web.stanford.edu/~jurafsky/slp3/
[zero-to-hero]: https://karpathy.ai/zero-to-hero.html

## Kernels inside a transformer block

Each paper defines an operation that maps to a PMPP.4e pattern.

- [Online normalizer calculation for softmax][online-softmax]: one-pass max
  and sum, the reduction trick that makes fused attention possible.
- [FlashAttention][flash1], [FlashAttention-2][flash2], and
  [FlashAttention-3][flash3]: tiled attention that never materializes the
  score matrix. GEMM tiling, online softmax, and asynchronous pipelines in one
  kernel.
- [Layer Normalization][layernorm] and [RMSNorm][rmsnorm]: per-row
  reductions fused with elementwise scaling.
- [RoFormer][rope]: rotary position embeddings, an elementwise kernel on
  queries and keys.
- [GQA][gqa]: grouped-query attention; fewer KV heads shrink the KV cache and
  change attention kernel shapes.
- [LLaMA][llama] and [Llama 2][llama2]: the block most open models use:
  RMSNorm, RoPE, SwiGLU, and GQA.

[flash1]: https://arxiv.org/abs/2205.14135
[flash2]: https://arxiv.org/abs/2307.08691
[flash3]: https://arxiv.org/abs/2407.08608
[gqa]: https://arxiv.org/abs/2305.13245
[layernorm]: https://arxiv.org/abs/1607.06450
[llama]: https://arxiv.org/abs/2302.13971
[llama2]: https://arxiv.org/abs/2307.09288
[online-softmax]: https://arxiv.org/abs/1805.02867
[rmsnorm]: https://arxiv.org/abs/1910.07467
[rope]: https://arxiv.org/abs/2104.09864

## Training at scale

- [How to Scale Your Model][scaling-book]: rooflines, memory and compute
  arithmetic, and the parallelism strategies, written for TPUs but
  hardware-independent in its reasoning.
- [The Ultra-Scale Playbook][ultrascale]: data, tensor, pipeline, context, and
  expert parallelism on GPU clusters, with measured results.
- [Megatron-LM][megatron]: tensor parallelism for transformer layers.
- [ZeRO][zero]: sharding optimizer state, gradients, and parameters.
- [GPipe][gpipe]: pipeline parallelism with micro-batches.
- [Scaling Laws for Neural Language Models][kaplan] and [Training
  Compute-Optimal Large Language Models][chinchilla]: how loss scales with
  parameters, data, and compute.
- [Switch Transformers][switch] and [GShard][gshard]: mixture of experts;
  routing turns dense GEMM into grouped GEMM plus all-to-all communication.
- [NCCL][nccl] in `cuda.md` covers the collectives every strategy above
  relies on.

[chinchilla]: https://arxiv.org/abs/2203.15556
[gpipe]: https://arxiv.org/abs/1811.06965
[gshard]: https://arxiv.org/abs/2006.16668
[kaplan]: https://arxiv.org/abs/2001.08361
[megatron]: https://arxiv.org/abs/1909.08053
[nccl]: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/
[scaling-book]: https://jax-ml.github.io/scaling-book/
[switch]: https://arxiv.org/abs/2101.03961
[ultrascale]: https://huggingface.co/spaces/nanotron/ultrascale-playbook
[zero]: https://arxiv.org/abs/1910.02054

## Inference

- [Transformer Inference Arithmetic][inference-arithmetic]: KV cache size,
  memory-bound decode, latency and throughput estimates from first principles.
- [Efficient Memory Management for LLM Serving with
  PagedAttention][paged-attention]: the vLLM design; KV cache as paged
  memory.
- [Fast Inference from Transformers via Speculative Decoding][speculative]:
  draft and verify to trade compute for latency.
- [TensorRT LLM][tensorrt-llm] in `cuda.md` is the NVIDIA implementation of
  these ideas.

[inference-arithmetic]: https://kipp.ly/p/transformer-inference-arithmetic
[paged-attention]: https://arxiv.org/abs/2309.06180
[speculative]: https://arxiv.org/abs/2211.17192
[tensorrt-llm]: https://nvidia.github.io/TensorRT-LLM/

## Quantization and low precision

Histogram, reduction, and elementwise kernels decide the scales; GEMM kernels
consume the packed formats.

- [FP8 Formats for Deep Learning][fp8]: E4M3 and E5M2, the formats Tensor
  Cores and Transformer Engine use.
- [LLM.int8()][llm-int8]: outlier features and mixed-precision decomposition.
- [SmoothQuant][smoothquant]: migrating activation outliers into weights for
  W8A8.
- [GPTQ][gptq] and [AWQ][awq]: weight-only post-training quantization with
  calibration data.

[awq]: https://arxiv.org/abs/2306.00978
[fp8]: https://arxiv.org/abs/2209.05433
[gptq]: https://arxiv.org/abs/2210.17323
[llm-int8]: https://arxiv.org/abs/2208.07339
[smoothquant]: https://arxiv.org/abs/2211.10438
