# Dot Product

## Overview

Dot-product kernels (scalar and vectorized variants).

- [x] dot_product — FP32 / FP16
- [x] dot_product_fp32x4 — vectorized FP32
- [x] dot_product_fp16x2 — vectorized FP16
- [x] dot_product_fp16x8 — vectorized FP16, packed r/w
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
torch                          mean time: 0.037600 ms
dot_product                    mean time: 0.045568 ms, speedup: 0.83
dot_product_fp32x4             mean time: 0.053104 ms, speedup: 0.71
torch                          mean time: 0.025648 ms
dot_product                    mean time: 0.041696 ms, speedup: 0.62
dot_product_fp16x2             mean time: 0.028704 ms, speedup: 0.89
dot_product_fp16x8_packed      mean time: 0.029376 ms, speedup: 0.87
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.059488 ms
dot_product                    mean time: 0.071280 ms, speedup: 0.83
dot_product_fp32x4             mean time: 0.067024 ms, speedup: 0.89
torch                          mean time: 0.036784 ms
dot_product                    mean time: 0.064608 ms, speedup: 0.57
dot_product_fp16x2             mean time: 0.047360 ms, speedup: 0.78
dot_product_fp16x8_packed      mean time: 0.053776 ms, speedup: 0.68
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.108720 ms
dot_product                    mean time: 0.120128 ms, speedup: 0.91
dot_product_fp32x4             mean time: 0.128256 ms, speedup: 0.85
torch                          mean time: 0.061952 ms
dot_product                    mean time: 0.090496 ms, speedup: 0.68
dot_product_fp16x2             mean time: 0.082512 ms, speedup: 0.75
dot_product_fp16x8_packed      mean time: 0.073552 ms, speedup: 0.84
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.060176 ms
dot_product                    mean time: 0.077376 ms, speedup: 0.78
dot_product_fp32x4             mean time: 0.070384 ms, speedup: 0.85
torch                          mean time: 0.036768 ms
dot_product                    mean time: 0.063952 ms, speedup: 0.57
dot_product_fp16x2             mean time: 0.047280 ms, speedup: 0.78
dot_product_fp16x8_packed      mean time: 0.052144 ms, speedup: 0.71
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.108384 ms
dot_product                    mean time: 0.121664 ms, speedup: 0.89
dot_product_fp32x4             mean time: 0.128464 ms, speedup: 0.84
torch                          mean time: 0.061856 ms
dot_product                    mean time: 0.090528 ms, speedup: 0.68
dot_product_fp16x2             mean time: 0.080992 ms, speedup: 0.76
dot_product_fp16x8_packed      mean time: 0.072512 ms, speedup: 0.85
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.209552 ms
dot_product                    mean time: 0.223408 ms, speedup: 0.94
dot_product_fp32x4             mean time: 0.223312 ms, speedup: 0.94
torch                          mean time: 0.113056 ms
dot_product                    mean time: 0.181024 ms, speedup: 0.62
dot_product_fp16x2             mean time: 0.124512 ms, speedup: 0.91
dot_product_fp16x8_packed      mean time: 0.123984 ms, speedup: 0.91
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.107712 ms
dot_product                    mean time: 0.123280 ms, speedup: 0.87
dot_product_fp32x4             mean time: 0.115728 ms, speedup: 0.93
torch                          mean time: 0.061936 ms
dot_product                    mean time: 0.090608 ms, speedup: 0.68
dot_product_fp16x2             mean time: 0.080512 ms, speedup: 0.77
dot_product_fp16x8_packed      mean time: 0.071904 ms, speedup: 0.86
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.209472 ms
dot_product                    mean time: 0.226800 ms, speedup: 0.92
dot_product_fp32x4             mean time: 0.215968 ms, speedup: 0.97
torch                          mean time: 0.113264 ms
dot_product                    mean time: 0.174976 ms, speedup: 0.65
dot_product_fp16x2             mean time: 0.128368 ms, speedup: 0.88
dot_product_fp16x8_packed      mean time: 0.128128 ms, speedup: 0.88
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.403200 ms
dot_product                    mean time: 0.418528 ms, speedup: 0.96
dot_product_fp32x4             mean time: 0.407856 ms, speedup: 0.99
torch                          mean time: 0.214192 ms
dot_product                    mean time: 0.279984 ms, speedup: 0.77
dot_product_fp16x2             mean time: 0.235008 ms, speedup: 0.91
dot_product_fp16x8_packed      mean time: 0.220992 ms, speedup: 0.97
```
