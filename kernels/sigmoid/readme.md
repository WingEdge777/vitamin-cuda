# Sigmoid

## Overview

Sigmoid kernels.

- [x] sigmoid — FP32 / FP16
- [x] sigmoid_fp16x2 — vectorized FP16
- [x] sigmoid_fp16x8 — vectorized FP16
- [x] sigmoid_fp16x8_packed — vectorized FP16, packed r/w
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
torch                          mean time: 0.031808 ms
sigmoid                        mean time: 0.042128 ms, speedup: 0.76
sigmoid_fp32x4                 mean time: 0.029504 ms, speedup: 1.08
torch                          mean time: 0.015920 ms
sigmoid_half                   mean time: 0.032288 ms, speedup: 0.49
sigmoid_fp16x2                 mean time: 0.019264 ms, speedup: 0.83
sigmoid_fp16x8                 mean time: 0.015520 ms, speedup: 1.03
sigmoid_fp16x8_packed          mean time: 0.013184 ms, speedup: 1.21
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.051392 ms
sigmoid                        mean time: 0.062176 ms, speedup: 0.83
sigmoid_fp32x4                 mean time: 0.051952 ms, speedup: 0.99
torch                          mean time: 0.023392 ms
sigmoid_half                   mean time: 0.050784 ms, speedup: 0.46
sigmoid_fp16x2                 mean time: 0.033888 ms, speedup: 0.69
sigmoid_fp16x8                 mean time: 0.031984 ms, speedup: 0.73
sigmoid_fp16x8_packed          mean time: 0.023376 ms, speedup: 1.00
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.105568 ms
sigmoid                        mean time: 0.119664 ms, speedup: 0.88
sigmoid_fp32x4                 mean time: 0.104672 ms, speedup: 1.01
torch                          mean time: 0.054496 ms
sigmoid_half                   mean time: 0.091968 ms, speedup: 0.59
sigmoid_fp16x2                 mean time: 0.062496 ms, speedup: 0.87
sigmoid_fp16x8                 mean time: 0.058016 ms, speedup: 0.94
sigmoid_fp16x8_packed          mean time: 0.050528 ms, speedup: 1.08
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.052688 ms
sigmoid                        mean time: 0.066160 ms, speedup: 0.80
sigmoid_fp32x4                 mean time: 0.053056 ms, speedup: 0.99
torch                          mean time: 0.026224 ms
sigmoid_half                   mean time: 0.058112 ms, speedup: 0.45
sigmoid_fp16x2                 mean time: 0.040320 ms, speedup: 0.65
sigmoid_fp16x8                 mean time: 0.037472 ms, speedup: 0.70
sigmoid_fp16x8_packed          mean time: 0.026208 ms, speedup: 1.00
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.103552 ms
sigmoid                        mean time: 0.133424 ms, speedup: 0.78
sigmoid_fp32x4                 mean time: 0.104592 ms, speedup: 0.99
torch                          mean time: 0.052480 ms
sigmoid_half                   mean time: 0.094240 ms, speedup: 0.56
sigmoid_fp16x2                 mean time: 0.063520 ms, speedup: 0.83
sigmoid_fp16x8                 mean time: 0.060448 ms, speedup: 0.87
sigmoid_fp16x8_packed          mean time: 0.053600 ms, speedup: 0.98
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.207664 ms
sigmoid                        mean time: 0.231360 ms, speedup: 0.90
sigmoid_fp32x4                 mean time: 0.209184 ms, speedup: 0.99
torch                          mean time: 0.105920 ms
sigmoid_half                   mean time: 0.171360 ms, speedup: 0.62
sigmoid_fp16x2                 mean time: 0.118352 ms, speedup: 0.89
sigmoid_fp16x8                 mean time: 0.108160 ms, speedup: 0.98
sigmoid_fp16x8_packed          mean time: 0.104896 ms, speedup: 1.01
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.104448 ms
sigmoid                        mean time: 0.117728 ms, speedup: 0.89
sigmoid_fp32x4                 mean time: 0.105680 ms, speedup: 0.99
torch                          mean time: 0.053072 ms
sigmoid_half                   mean time: 0.094080 ms, speedup: 0.56
sigmoid_fp16x2                 mean time: 0.062688 ms, speedup: 0.85
sigmoid_fp16x8                 mean time: 0.059312 ms, speedup: 0.89
sigmoid_fp16x8_packed          mean time: 0.050944 ms, speedup: 1.04
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.205712 ms
sigmoid                        mean time: 0.240112 ms, speedup: 0.86
sigmoid_fp32x4                 mean time: 0.208848 ms, speedup: 0.98
torch                          mean time: 0.105376 ms
sigmoid_half                   mean time: 0.188224 ms, speedup: 0.56
sigmoid_fp16x2                 mean time: 0.131136 ms, speedup: 0.80
sigmoid_fp16x8                 mean time: 0.114288 ms, speedup: 0.92
sigmoid_fp16x8_packed          mean time: 0.105856 ms, speedup: 1.00
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.407056 ms
sigmoid                        mean time: 0.433024 ms, speedup: 0.94
sigmoid_fp32x4                 mean time: 0.404896 ms, speedup: 1.01
torch                          mean time: 0.208928 ms
sigmoid_half                   mean time: 0.325456 ms, speedup: 0.64
sigmoid_fp16x2                 mean time: 0.229472 ms, speedup: 0.91
sigmoid_fp16x8                 mean time: 0.212912 ms, speedup: 0.98
sigmoid_fp16x8_packed          mean time: 0.206304 ms, speedup: 1.01
```
