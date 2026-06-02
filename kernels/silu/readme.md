# silu

## Overview

SiLU (Swish) kernels.

- [x] silu — FP32 / FP16
- [x] silu_fp16x2 — vectorized FP16
- [x] silu_fp16x8 — vectorized FP16
- [x] silu_fp16x8_packed — vectorized FP16, packed r/w
- [x] pytorch op bindings && diff check

## Run tests

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### Sample output

```bash
####################################################################################################
n: 1024, m: 1024
torch                          mean time: 0.028128 ms
silu                           mean time: 0.037760 ms, speedup: 0.74
silu_fp32x4                    mean time: 0.023744 ms, speedup: 1.18
torch                          mean time: 0.017568 ms
silu_half                      mean time: 0.028656 ms, speedup: 0.61
silu_fp16x2                    mean time: 0.020832 ms, speedup: 0.84
silu_fp16x8                    mean time: 0.016560 ms, speedup: 1.06
silu_fp16x8_packed             mean time: 0.013456 ms, speedup: 1.31
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.052512 ms
silu                           mean time: 0.061664 ms, speedup: 0.85
silu_fp32x4                    mean time: 0.052544 ms, speedup: 1.00
torch                          mean time: 0.025472 ms
silu_half                      mean time: 0.050416 ms, speedup: 0.51
silu_fp16x2                    mean time: 0.033920 ms, speedup: 0.75
silu_fp16x8                    mean time: 0.031248 ms, speedup: 0.82
silu_fp16x8_packed             mean time: 0.023440 ms, speedup: 1.09
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.100160 ms
silu                           mean time: 0.117312 ms, speedup: 0.85
silu_fp32x4                    mean time: 0.104800 ms, speedup: 0.96
torch                          mean time: 0.049792 ms
silu_half                      mean time: 0.089088 ms, speedup: 0.56
silu_fp16x2                    mean time: 0.061552 ms, speedup: 0.81
silu_fp16x8                    mean time: 0.057024 ms, speedup: 0.87
silu_fp16x8_packed             mean time: 0.050928 ms, speedup: 0.98
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.048288 ms
silu                           mean time: 0.061952 ms, speedup: 0.78
silu_fp32x4                    mean time: 0.052320 ms, speedup: 0.92
torch                          mean time: 0.025776 ms
silu_half                      mean time: 0.050128 ms, speedup: 0.51
silu_fp16x2                    mean time: 0.034176 ms, speedup: 0.75
silu_fp16x8                    mean time: 0.031312 ms, speedup: 0.82
silu_fp16x8_packed             mean time: 0.023552 ms, speedup: 1.09
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.099248 ms
silu                           mean time: 0.118000 ms, speedup: 0.84
silu_fp32x4                    mean time: 0.104224 ms, speedup: 0.95
torch                          mean time: 0.049648 ms
silu_half                      mean time: 0.092912 ms, speedup: 0.53
silu_fp16x2                    mean time: 0.062720 ms, speedup: 0.79
silu_fp16x8                    mean time: 0.060096 ms, speedup: 0.83
silu_fp16x8_packed             mean time: 0.054016 ms, speedup: 0.92
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.203888 ms
silu                           mean time: 0.232784 ms, speedup: 0.88
silu_fp32x4                    mean time: 0.208960 ms, speedup: 0.98
torch                          mean time: 0.099952 ms
silu_half                      mean time: 0.171680 ms, speedup: 0.58
silu_fp16x2                    mean time: 0.118368 ms, speedup: 0.84
silu_fp16x8                    mean time: 0.109200 ms, speedup: 0.92
silu_fp16x8_packed             mean time: 0.105152 ms, speedup: 0.95
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.100416 ms
silu                           mean time: 0.117888 ms, speedup: 0.85
silu_fp32x4                    mean time: 0.106016 ms, speedup: 0.95
torch                          mean time: 0.049984 ms
silu_half                      mean time: 0.089440 ms, speedup: 0.56
silu_fp16x2                    mean time: 0.061104 ms, speedup: 0.82
silu_fp16x8                    mean time: 0.058224 ms, speedup: 0.86
silu_fp16x8_packed             mean time: 0.051808 ms, speedup: 0.96
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.202368 ms
silu                           mean time: 0.231408 ms, speedup: 0.87
silu_fp32x4                    mean time: 0.208544 ms, speedup: 0.97
torch                          mean time: 0.099840 ms
silu_half                      mean time: 0.177616 ms, speedup: 0.56
silu_fp16x2                    mean time: 0.120464 ms, speedup: 0.83
silu_fp16x8                    mean time: 0.109776 ms, speedup: 0.91
silu_fp16x8_packed             mean time: 0.105840 ms, speedup: 0.94
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.404112 ms
silu                           mean time: 0.431648 ms, speedup: 0.94
silu_fp32x4                    mean time: 0.404432 ms, speedup: 1.00
torch                          mean time: 0.202048 ms
silu_half                      mean time: 0.321248 ms, speedup: 0.63
silu_fp16x2                    mean time: 0.229792 ms, speedup: 0.88
silu_fp16x8                    mean time: 0.211008 ms, speedup: 0.96
silu_fp16x8_packed             mean time: 0.206448 ms, speedup: 0.98
```
