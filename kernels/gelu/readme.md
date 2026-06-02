# GELU

## Overview

GELU kernels.

- [x] gelu — FP32 / FP16
- [x] gelu_fp16x2 — vectorized FP16
- [x] gelu_fp16x8 — vectorized FP16
- [x] gelu_fp16x8_packed — vectorized FP16, packed r/w (~2× via `half2`)
- [x] pytorch op bindings && diff check

## Run tests

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

Torch’s GELU here had no preallocated-output fast path, so microbenchmarks favor our kernels. In tight loops the allocator noise fades and PyTorch can look surprisingly fast—worth revisiting apples-to-apples.

### Sample output

```bash
####################################################################################################
n: 1024, m: 1024
torch                          mean time: 0.025856 ms
gelu                           mean time: 0.058672 ms, speedup: 0.44
gelu_fp32x4                    mean time: 0.058112 ms, speedup: 0.44
torch                          mean time: 0.013280 ms
gelu_half                      mean time: 0.029008 ms, speedup: 0.46
gelu_fp16x2                    mean time: 0.019680 ms, speedup: 0.67
gelu_fp16x8                    mean time: 0.017184 ms, speedup: 0.77
gelu_fp16x8_packed             mean time: 0.013152 ms, speedup: 1.01
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.051248 ms
gelu                           mean time: 0.108128 ms, speedup: 0.47
gelu_fp32x4                    mean time: 0.108208 ms, speedup: 0.47
torch                          mean time: 0.023712 ms
gelu_half                      mean time: 0.051632 ms, speedup: 0.46
gelu_fp16x2                    mean time: 0.033888 ms, speedup: 0.70
gelu_fp16x8                    mean time: 0.031392 ms, speedup: 0.76
gelu_fp16x8_packed             mean time: 0.023232 ms, speedup: 1.02
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.101936 ms
gelu                           mean time: 0.213152 ms, speedup: 0.48
gelu_fp32x4                    mean time: 0.211312 ms, speedup: 0.48
torch                          mean time: 0.051680 ms
gelu_half                      mean time: 0.091008 ms, speedup: 0.57
gelu_fp16x2                    mean time: 0.062496 ms, speedup: 0.83
gelu_fp16x8                    mean time: 0.059264 ms, speedup: 0.87
gelu_fp16x8_packed             mean time: 0.052112 ms, speedup: 0.99
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.051920 ms
gelu                           mean time: 0.108144 ms, speedup: 0.48
gelu_fp32x4                    mean time: 0.108768 ms, speedup: 0.48
torch                          mean time: 0.023680 ms
gelu_half                      mean time: 0.052352 ms, speedup: 0.45
gelu_fp16x2                    mean time: 0.035712 ms, speedup: 0.66
gelu_fp16x8                    mean time: 0.032384 ms, speedup: 0.73
gelu_fp16x8_packed             mean time: 0.025120 ms, speedup: 0.94
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.132928 ms
gelu                           mean time: 0.219360 ms, speedup: 0.61
gelu_fp32x4                    mean time: 0.218928 ms, speedup: 0.61
torch                          mean time: 0.052048 ms
gelu_half                      mean time: 0.097104 ms, speedup: 0.54
gelu_fp16x2                    mean time: 0.064736 ms, speedup: 0.80
gelu_fp16x8                    mean time: 0.059072 ms, speedup: 0.88
gelu_fp16x8_packed             mean time: 0.051632 ms, speedup: 1.01
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.206048 ms
gelu                           mean time: 0.450112 ms, speedup: 0.46
gelu_fp32x4                    mean time: 0.423776 ms, speedup: 0.49
torch                          mean time: 0.102672 ms
gelu_half                      mean time: 0.177920 ms, speedup: 0.58
gelu_fp16x2                    mean time: 0.124752 ms, speedup: 0.82
gelu_fp16x8                    mean time: 0.110992 ms, speedup: 0.93
gelu_fp16x8_packed             mean time: 0.103376 ms, speedup: 0.99
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.101088 ms
gelu                           mean time: 0.226736 ms, speedup: 0.45
gelu_fp32x4                    mean time: 0.240160 ms, speedup: 0.42
torch                          mean time: 0.050608 ms
gelu_half                      mean time: 0.098880 ms, speedup: 0.51
gelu_fp16x2                    mean time: 0.065040 ms, speedup: 0.78
gelu_fp16x8                    mean time: 0.060112 ms, speedup: 0.84
gelu_fp16x8_packed             mean time: 0.051856 ms, speedup: 0.98
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.205088 ms
gelu                           mean time: 0.458576 ms, speedup: 0.45
gelu_fp32x4                    mean time: 0.448736 ms, speedup: 0.46
torch                          mean time: 0.103840 ms
gelu_half                      mean time: 0.176976 ms, speedup: 0.59
gelu_fp16x2                    mean time: 0.122416 ms, speedup: 0.85
gelu_fp16x8                    mean time: 0.107280 ms, speedup: 0.97
gelu_fp16x8_packed             mean time: 0.102960 ms, speedup: 1.01
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.406112 ms
gelu                           mean time: 0.831136 ms, speedup: 0.49
gelu_fp32x4                    mean time: 0.822448 ms, speedup: 0.49
torch                          mean time: 0.205120 ms
gelu_half                      mean time: 0.352656 ms, speedup: 0.58
gelu_fp16x2                    mean time: 0.231968 ms, speedup: 0.88
gelu_fp16x8                    mean time: 0.209296 ms, speedup: 0.98
gelu_fp16x8_packed             mean time: 0.204832 ms, speedup: 1.00
```
