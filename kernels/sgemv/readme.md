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
torch                          mean time: 0.013888 ms
gemv                           mean time: 0.013888 ms, speedup: 1.00
gemv_fp32x4                    mean time: 0.013344 ms, speedup: 1.04
####################################################################################################
n: 512, m: 1024
torch                          mean time: 0.015120 ms
gemv                           mean time: 0.020160 ms, speedup: 0.75
gemv_fp32x4                    mean time: 0.014432 ms, speedup: 1.05
####################################################################################################
n: 512, m: 2048
torch                          mean time: 0.020432 ms
gemv                           mean time: 0.032160 ms, speedup: 0.64
gemv_fp32x4                    mean time: 0.019648 ms, speedup: 1.04
####################################################################################################
n: 512, m: 4096
torch                          mean time: 0.033168 ms
gemv                           mean time: 0.057776 ms, speedup: 0.57
gemv_fp32x4                    mean time: 0.031760 ms, speedup: 1.04
####################################################################################################
n: 512, m: 8192
torch                          mean time: 0.072096 ms
gemv                           mean time: 0.108240 ms, speedup: 0.67
gemv_fp32x4                    mean time: 0.071728 ms, speedup: 1.01
####################################################################################################
n: 1024, m: 512
torch                          mean time: 0.017120 ms
gemv                           mean time: 0.019520 ms, speedup: 0.88
gemv_fp32x4                    mean time: 0.013696 ms, speedup: 1.25
####################################################################################################
n: 1024, m: 1024
torch                          mean time: 0.020976 ms
gemv                           mean time: 0.031968 ms, speedup: 0.66
gemv_fp32x4                    mean time: 0.019600 ms, speedup: 1.07
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.033552 ms
gemv                           mean time: 0.056448 ms, speedup: 0.59
gemv_fp32x4                    mean time: 0.031536 ms, speedup: 1.06
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.058032 ms
gemv                           mean time: 0.086480 ms, speedup: 0.67
gemv_fp32x4                    mean time: 0.056032 ms, speedup: 1.04
####################################################################################################
n: 1024, m: 8192
torch                          mean time: 0.106288 ms
gemv                           mean time: 0.167968 ms, speedup: 0.63
gemv_fp32x4                    mean time: 0.104896 ms, speedup: 1.01
####################################################################################################
n: 2048, m: 512
torch                          mean time: 0.021264 ms
gemv                           mean time: 0.032032 ms, speedup: 0.66
gemv_fp32x4                    mean time: 0.019648 ms, speedup: 1.08
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.033888 ms
gemv                           mean time: 0.054816 ms, speedup: 0.62
gemv_fp32x4                    mean time: 0.031600 ms, speedup: 1.07
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.059776 ms
gemv                           mean time: 0.081984 ms, speedup: 0.73
gemv_fp32x4                    mean time: 0.055680 ms, speedup: 1.07
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.105280 ms
gemv                           mean time: 0.156480 ms, speedup: 0.67
gemv_fp32x4                    mean time: 0.105424 ms, speedup: 1.00
####################################################################################################
n: 2048, m: 8192
torch                          mean time: 0.202240 ms
gemv                           mean time: 0.272128 ms, speedup: 0.74
gemv_fp32x4                    mean time: 0.204928 ms, speedup: 0.99
####################################################################################################
n: 4096, m: 512
torch                          mean time: 0.033120 ms
gemv                           mean time: 0.055696 ms, speedup: 0.59
gemv_fp32x4                    mean time: 0.031680 ms, speedup: 1.05
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.058848 ms
gemv                           mean time: 0.081152 ms, speedup: 0.73
gemv_fp32x4                    mean time: 0.056032 ms, speedup: 1.05
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.108752 ms
gemv                           mean time: 0.154256 ms, speedup: 0.71
gemv_fp32x4                    mean time: 0.105184 ms, speedup: 1.03
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.205696 ms
gemv                           mean time: 0.269696 ms, speedup: 0.76
gemv_fp32x4                    mean time: 0.207088 ms, speedup: 0.99
####################################################################################################
n: 4096, m: 8192
torch                          mean time: 0.394272 ms
gemv                           mean time: 0.464528 ms, speedup: 0.85
gemv_fp32x4                    mean time: 0.399488 ms, speedup: 0.99
####################################################################################################
n: 8192, m: 512
torch                          mean time: 0.059136 ms
gemv                           mean time: 0.082784 ms, speedup: 0.71
gemv_fp32x4                    mean time: 0.056128 ms, speedup: 1.05
####################################################################################################
n: 8192, m: 1024
torch                          mean time: 0.106704 ms
gemv                           mean time: 0.153504 ms, speedup: 0.70
gemv_fp32x4                    mean time: 0.106368 ms, speedup: 1.00
####################################################################################################
n: 8192, m: 2048
torch                          mean time: 0.214176 ms
gemv                           mean time: 0.267824 ms, speedup: 0.80
gemv_fp32x4                    mean time: 0.205568 ms, speedup: 1.04
####################################################################################################
n: 8192, m: 4096
torch                          mean time: 0.395120 ms
gemv                           mean time: 0.478128 ms, speedup: 0.83
gemv_fp32x4                    mean time: 0.407552 ms, speedup: 0.97
####################################################################################################
n: 8192, m: 8192
torch                          mean time: 0.808432 ms
gemv                           mean time: 0.878512 ms, speedup: 0.92
gemv_fp32x4                    mean time: 0.781808 ms, speedup: 1.03
```
