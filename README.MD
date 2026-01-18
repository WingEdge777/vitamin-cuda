# CUDA Practices

This repo is for practicing CUDA programming.

## Sample
  
## Kernels

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
- [relu](./kernels/relu)
  - [x] relu fp32/fp16 版
  - [x] relu_fp16x2(fp16向量化)
  - [x] relu_fp16x8(fp16向量化)
  - [x] relu_fp16x8(fp16向量化, packed r/w)
  - [x] pytorch op bindings && diff check
- [elu](./kernels/elu)
  - [x] elu fp32/fp16 版
  - [x] elu_fp16x2(fp16向量化)
  - [x] elu_fp16x8(fp16向量化)
  - [x] elu_fp16x8(fp16向量化, packed r/w, half2 近两倍提升)
  - [x] pytorch op bindings && diff check
- [gelu](./kernels/gelu) : 打不过 pytorch...
  - [x] gelu fp32/fp16 版
  - [x] gelu_fp16x2(fp16向量化)
  - [x] gelu_fp16x8(fp16向量化)
  - [x] gelu_fp16x8(fp16向量化，packed r/w)
- [x] pytorch op bindings && diff check

## Samples

- [deviceQuery](./samples/deviceQuery)

## Reference

- [LeetCUDA](https://github.com/xlite-dev/LeetCUDA)
- [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
