# GeMV

## Overview

GEMV (matrix–vector multiply) shows up whenever you map a feature vector to another dimension (wider or narrower). Batched GEMV is effectively a batched matmul, and most ML workloads use batching.

This folder keeps a few small reference kernels for practice.

- [x] gemv — FP32
- [x] gemv_fp32x4 — FP32 with vectorized loads
- [x] pytorch op bindings && diff check

## Run tests

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### Sample output

```bash
####################################################################################################
n: 512, m: 512
torch                          mean time: 0.008886 ms
gemv                           mean time: 0.009073 ms, speedup: 0.98
gemv_fp32x4                    mean time: 0.008982 ms, speedup: 0.99
####################################################################################################
n: 512, m: 1024
torch                          mean time: 0.020854 ms
gemv                           mean time: 0.011524 ms, speedup: 1.81
gemv_fp32x4                    mean time: 0.008784 ms, speedup: 2.37
####################################################################################################
n: 512, m: 2048
torch                          mean time: 0.012006 ms
gemv                           mean time: 0.013986 ms, speedup: 0.86
gemv_fp32x4                    mean time: 0.008806 ms, speedup: 1.36
####################################################################################################
n: 512, m: 4096
torch                          mean time: 0.030298 ms
gemv                           mean time: 0.013613 ms, speedup: 2.23
gemv_fp32x4                    mean time: 0.019828 ms, speedup: 1.53
####################################################################################################
n: 512, m: 8192
torch                          mean time: 0.060617 ms
gemv                           mean time: 0.021067 ms, speedup: 2.88
gemv_fp32x4                    mean time: 0.013917 ms, speedup: 4.36
####################################################################################################
n: 1024, m: 512
torch                          mean time: 0.038754 ms
gemv                           mean time: 0.021818 ms, speedup: 1.78
gemv_fp32x4                    mean time: 0.019504 ms, speedup: 1.99
####################################################################################################
n: 1024, m: 1024
torch                          mean time: 0.014897 ms
gemv                           mean time: 0.019693 ms, speedup: 0.76
gemv_fp32x4                    mean time: 0.009883 ms, speedup: 1.51
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.017987 ms
gemv                           mean time: 0.014623 ms, speedup: 1.23
gemv_fp32x4                    mean time: 0.022460 ms, speedup: 0.80
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.060782 ms
gemv                           mean time: 0.021470 ms, speedup: 2.83
gemv_fp32x4                    mean time: 0.013789 ms, speedup: 4.41
####################################################################################################
n: 1024, m: 8192
torch                          mean time: 0.033910 ms
gemv                           mean time: 0.049395 ms, speedup: 0.69
gemv_fp32x4                    mean time: 0.023526 ms, speedup: 1.44
####################################################################################################
n: 2048, m: 512
torch                          mean time: 0.021050 ms
gemv                           mean time: 0.012869 ms, speedup: 1.64
gemv_fp32x4                    mean time: 0.011093 ms, speedup: 1.90
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.027313 ms
gemv                           mean time: 0.020678 ms, speedup: 1.32
gemv_fp32x4                    mean time: 0.011111 ms, speedup: 2.46
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.015484 ms
gemv                           mean time: 0.022956 ms, speedup: 0.67
gemv_fp32x4                    mean time: 0.021555 ms, speedup: 0.72
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.110697 ms
gemv                           mean time: 0.044053 ms, speedup: 2.51
gemv_fp32x4                    mean time: 0.037044 ms, speedup: 2.99
####################################################################################################
n: 2048, m: 8192
torch                          mean time: 0.190447 ms
gemv                           mean time: 0.217329 ms, speedup: 0.88
gemv_fp32x4                    mean time: 0.207049 ms, speedup: 0.92
####################################################################################################
n: 4096, m: 512
torch                          mean time: 0.021090 ms
gemv                           mean time: 0.017069 ms, speedup: 1.24
gemv_fp32x4                    mean time: 0.012399 ms, speedup: 1.70
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.057493 ms
gemv                           mean time: 0.022998 ms, speedup: 2.50
gemv_fp32x4                    mean time: 0.022899 ms, speedup: 2.51
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.025669 ms
gemv                           mean time: 0.050391 ms, speedup: 0.51
gemv_fp32x4                    mean time: 0.034293 ms, speedup: 0.75
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.191087 ms
gemv                           mean time: 0.221770 ms, speedup: 0.86
gemv_fp32x4                    mean time: 0.206958 ms, speedup: 0.92
####################################################################################################
n: 4096, m: 8192
torch                          mean time: 0.375710 ms
gemv                           mean time: 0.414233 ms, speedup: 0.91
gemv_fp32x4                    mean time: 0.457492 ms, speedup: 0.82
####################################################################################################
n: 8192, m: 512
torch                          mean time: 0.028225 ms
gemv                           mean time: 0.034232 ms, speedup: 0.82
gemv_fp32x4                    mean time: 0.017243 ms, speedup: 1.64
####################################################################################################
n: 8192, m: 1024
torch                          mean time: 0.139370 ms
gemv                           mean time: 0.048529 ms, speedup: 2.87
gemv_fp32x4                    mean time: 0.026932 ms, speedup: 5.17
####################################################################################################
n: 8192, m: 2048
torch                          mean time: 0.192962 ms
gemv                           mean time: 0.216880 ms, speedup: 0.89
gemv_fp32x4                    mean time: 0.207960 ms, speedup: 0.93
####################################################################################################
n: 8192, m: 4096
torch                          mean time: 0.373298 ms
gemv                           mean time: 0.406433 ms, speedup: 0.92
gemv_fp32x4                    mean time: 0.388672 ms, speedup: 0.96
####################################################################################################
n: 8192, m: 8192
torch                          mean time: 0.738979 ms
gemv                           mean time: 0.775295 ms, speedup: 0.95
gemv_fp32x4                    mean time: 0.753339 ms, speedup: 0.98
```
