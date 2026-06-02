# hardswish

## Overview

Hardswish kernels.

- [x] hardswish — FP32 / FP16
- [x] hardswish_fp16x2 — vectorized FP16
- [x] hardswish_fp16x8 — vectorized FP16
- [x] hardswish_fp16x8_packed — vectorized FP16, packed r/w
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
torch                          mean time: 0.027616 ms
hardswish                      mean time: 0.036224 ms, speedup: 0.76
hardswish_fp32x4               mean time: 0.025792 ms, speedup: 1.07
torch                          mean time: 0.014672 ms
hardswish_half                 mean time: 0.028896 ms, speedup: 0.51
hardswish_fp16x2               mean time: 0.020880 ms, speedup: 0.70
hardswish_fp16x8               mean time: 0.013408 ms, speedup: 1.09
hardswish_fp16x8_packed        mean time: 0.013200 ms, speedup: 1.11
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.049104 ms
hardswish                      mean time: 0.060176 ms, speedup: 0.82
hardswish_fp32x4               mean time: 0.052752 ms, speedup: 0.93
torch                          mean time: 0.023696 ms
hardswish_half                 mean time: 0.049360 ms, speedup: 0.48
hardswish_fp16x2               mean time: 0.033952 ms, speedup: 0.70
hardswish_fp16x8               mean time: 0.027312 ms, speedup: 0.87
hardswish_fp16x8_packed        mean time: 0.023488 ms, speedup: 1.01
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.101328 ms
hardswish                      mean time: 0.115744 ms, speedup: 0.88
hardswish_fp32x4               mean time: 0.103520 ms, speedup: 0.98
torch                          mean time: 0.046336 ms
hardswish_half                 mean time: 0.087264 ms, speedup: 0.53
hardswish_fp16x2               mean time: 0.061280 ms, speedup: 0.76
hardswish_fp16x8               mean time: 0.055824 ms, speedup: 0.83
hardswish_fp16x8_packed        mean time: 0.052560 ms, speedup: 0.88
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.048832 ms
hardswish                      mean time: 0.060064 ms, speedup: 0.81
hardswish_fp32x4               mean time: 0.050528 ms, speedup: 0.97
torch                          mean time: 0.023024 ms
hardswish_half                 mean time: 0.048384 ms, speedup: 0.48
hardswish_fp16x2               mean time: 0.033664 ms, speedup: 0.68
hardswish_fp16x8               mean time: 0.027424 ms, speedup: 0.84
hardswish_fp16x8_packed        mean time: 0.023600 ms, speedup: 0.98
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.100768 ms
hardswish                      mean time: 0.114112 ms, speedup: 0.88
hardswish_fp32x4               mean time: 0.105440 ms, speedup: 0.96
torch                          mean time: 0.046112 ms
hardswish_half                 mean time: 0.086736 ms, speedup: 0.53
hardswish_fp16x2               mean time: 0.061040 ms, speedup: 0.76
hardswish_fp16x8               mean time: 0.053360 ms, speedup: 0.86
hardswish_fp16x8_packed        mean time: 0.052096 ms, speedup: 0.89
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.203008 ms
hardswish                      mean time: 0.225024 ms, speedup: 0.90
hardswish_fp32x4               mean time: 0.203728 ms, speedup: 1.00
torch                          mean time: 0.101424 ms
hardswish_half                 mean time: 0.162784 ms, speedup: 0.62
hardswish_fp16x2               mean time: 0.115248 ms, speedup: 0.88
hardswish_fp16x8               mean time: 0.103520 ms, speedup: 0.98
hardswish_fp16x8_packed        mean time: 0.102352 ms, speedup: 0.99
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.100896 ms
hardswish                      mean time: 0.115520 ms, speedup: 0.87
hardswish_fp32x4               mean time: 0.102848 ms, speedup: 0.98
torch                          mean time: 0.046064 ms
hardswish_half                 mean time: 0.087520 ms, speedup: 0.53
hardswish_fp16x2               mean time: 0.061792 ms, speedup: 0.75
hardswish_fp16x8               mean time: 0.054832 ms, speedup: 0.84
hardswish_fp16x8_packed        mean time: 0.068608 ms, speedup: 0.67
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.203840 ms
hardswish                      mean time: 0.227024 ms, speedup: 0.90
hardswish_fp32x4               mean time: 0.205344 ms, speedup: 0.99
torch                          mean time: 0.101456 ms
hardswish_half                 mean time: 0.192768 ms, speedup: 0.53
hardswish_fp16x2               mean time: 0.145088 ms, speedup: 0.70
hardswish_fp16x8               mean time: 0.136368 ms, speedup: 0.74
hardswish_fp16x8_packed        mean time: 0.135616 ms, speedup: 0.75
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.516928 ms
hardswish                      mean time: 0.423056 ms, speedup: 1.22
hardswish_fp32x4               mean time: 0.401536 ms, speedup: 1.29
torch                          mean time: 0.202944 ms
hardswish_half                 mean time: 0.314096 ms, speedup: 0.65
hardswish_fp16x2               mean time: 0.227440 ms, speedup: 0.89
hardswish_fp16x8               mean time: 0.206576 ms, speedup: 0.98
hardswish_fp16x8_packed        mean time: 0.205008 ms, speedup: 0.99
```
