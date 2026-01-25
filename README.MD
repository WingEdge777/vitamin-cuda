# Vitamin-CUDA

> **One Kernel a Day, Keeps High Latency Away.** 🚀

Welcome to your daily dose of CUDA programming! **Vitamin-CUDA** is a curated collection of hands-on CUDA practices, designed to take you from **Hello World** to **High Performance**. Whether you are a beginner looking to understand the grid-stride loop or an enthusiast diving into warp-level primitives, there's a kernel here for you.

💊Let's get started and happy coding! 🧠

## Prerequisites 🛠️

- NVIDIA GPU (Compute Capability 6.0+)
- CUDA Toolkit 11.0+
- C++ Compiler (GCC/Clang/MSVC)
- CMake 3.18+ (Optional, but recommended)
- PyTorch (For extension examples/python bingding and performence comparation)

I recommend using nvidia pytorch ngc docker images for a quick start! Refer to <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch>

## Kernels (40+ kernels)

All kernels were tested on an RTX 5060 GPU and benchmarked against PyTorch 2.9

### Easy (🌟~🌟🌟)

- [elementwise](./kernels/elementwise): elementwise add
  - [x] elementwise_add fp32/fp16 版
  - [x] elementwise_add_fp16x2(fp16向量化)
  - [x] elementwise_add_fp16x8(fp16向量化)
  - [x] elementwise_add_fp16x8(fp16向量化, packed r/w)
  - [x] pytorch op bindings && diff check
- [sigmoid](./kernels/sigmoid)
  - [x] sigmoid fp32/fp16 版
  - [x] sigmoid_fp16x2(fp16向量化)
  - [x] sigmoid_fp16x8(fp16向量化)
  - [x] sigmoid_fp16x8(fp16向量化, packed r/w)
  - [x] pytorch op bindings && diff check
- [swish](./kernels/swish)
  - [x] swish fp32/fp16 版
  - [x] swish_fp16x2(fp16向量化)
  - [x] swish_fp16x8(fp16向量化)
  - [x] swish_fp16x8(fp16向量化, packed r/w)
  - [x] pytorch op bindings && diff check
- [relu](./kernels/relu)
  - [x] relu fp32/fp16 版
  - [x] relu_fp16x2(fp16向量化)
  - [x] relu_fp16x8(fp16向量化)
  - [x] relu_fp16x8(fp16向量化, packed r/w)
  - [x] pytorch op bindings && diff check
- [relu6](./kernels/relu6)
  - [x] relu6 fp32/fp16 版
  - [x] relu6_fp16x2(fp16向量化)
  - [x] relu6_fp16x8(fp16向量化)
  - [x] relu6_fp16x8(fp16向量化, packed r/w)
  - [x] pytorch op bindings && diff check
- [elu](./kernels/elu)
  - [x] elu fp32/fp16 版
  - [x] elu_fp16x2(fp16向量化)
  - [x] elu_fp16x8(fp16向量化)
  - [x] elu_fp16x8(fp16向量化, packed r/w, half2 近两倍提升)
  - [x] pytorch op bindings && diff check
- [gelu](./kernels/gelu)
  - [x] gelu fp32/fp16 版
  - [x] gelu_fp16x2(fp16向量化)
  - [x] gelu_fp16x8(fp16向量化)
  - [x] gelu_fp16x8(fp16向量化，packed r/w)
  - [x] pytorch op bindings && diff check
- [hardswish](./kernels/hardswish)
  - [x] hardswish fp32/fp16 版
  - [x] hardswish_fp16x2(fp16向量化)
  - [x] hardswish_fp16x8(fp16向量化)
  - [x] hardswish_fp16x8(fp16向量化, packed r/w)
  - [x] pytorch op bindings && diff check
- [embedding](./kernels/embedding/)
  - [x] embedding fp32/fp16 版
  - [x] embedding_fp32x4(fp32向量化)
  - [x] embedding_fp32x4(fp32向量化, packed r/w)
  - [x] embedding_fp16x2(fp16向量化)
  - [x] embedding_fp16x8(fp16向量化)
  - [x] embedding_fp16x8(fp16向量化, packed r/w)
  - [x] pytorch op bindings && diff check

### Medium (🌟🌟~🌟🌟🌟)

To be continued...

### Hard (🌟🌟🌟~🌟🌟🌟🌟)

To be continued...

## Samples

- [deviceQuery](./samples/deviceQuery)

## Reference

- [LeetCUDA](https://github.com/xlite-dev/LeetCUDA)
- [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
