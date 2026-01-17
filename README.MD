# CUDA Practices

This repo is for practicing CUDA programming.

## Sample
  
## Kernels

- [elementwise](./kernels/elementwise): elementwise add
  - [x] elementwise_add fp32/fp16 版
  - [x] elementwise_add_fp16x2(fp16向量化)
  - [x] elementwise_add_fp16x8(fp16向量化)
  - [x] elementwise_add_fp16x8(fp16向量化, packed r/w)
  - [x] pytorch op bindings
- [sigmoid](./kernels/sigmoid)
  - [x] sigmoid fp32/fp16 版
  - [x] sigmoid_fp16x2(fp16向量化)
  - [x] sigmoid_fp16x8(fp16向量化)
  - [x] sigmoid_fp16x8(fp16向量化, packed r/w)
  - [x] pytorch op bindings
- [relu](./kernels/relu)
  - [x] relu fp32/fp16 版
  - [x] relu_fp16x2(fp16向量化)
  - [x] relu_fp16x8(fp16向量化)
  - [x] relu_fp16x8(fp16向量化, packed r/w)
  - [x] pytorch op bindings
- [elu](./kernels/elu)
  - [x] elu fp32/fp16 版
  - [x] elu_fp16x2(fp16向量化)
  - [x] elu_fp16x8(fp16向量化)
  - [x] elu_fp16x8(fp16向量化, packeds r/w, half2 近两倍提升)
  - [x] pytorch op bindings

## Reference

- [LeetCUDA](https://github.com/xlite-dev/LeetCUDA)
- [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
