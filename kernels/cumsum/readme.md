# cumsum

## Overview

cumsum kernels.

- [x] naive Torch cumsum
- [x] cumsum — FP32
- [x] cumsum — FP32x4
- [x] pytorch op bindings && diff check

## Run tests

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### Sample output

```bash
####################################################################################################
n: 1, m: 2048
torch                          mean time: 0.010816 ms
cumsum_fp32                    mean time: 0.011504 ms, speedup: 0.94
cumsum_fp32x4                  mean time: 0.007072 ms, speedup: 1.53
####################################################################################################
n: 1, m: 4096
torch                          mean time: 0.010752 ms
cumsum_fp32                    mean time: 0.016576 ms, speedup: 0.65
cumsum_fp32x4                  mean time: 0.008848 ms, speedup: 1.22
####################################################################################################
n: 1, m: 8192
torch                          mean time: 0.010752 ms
cumsum_fp32                    mean time: 0.025712 ms, speedup: 0.42
cumsum_fp32x4                  mean time: 0.011296 ms, speedup: 0.95
####################################################################################################
n: 1, m: 12800
torch                          mean time: 0.010976 ms
cumsum_fp32                    mean time: 0.037344 ms, speedup: 0.29
cumsum_fp32x4                  mean time: 0.015120 ms, speedup: 0.73
####################################################################################################
n: 32, m: 2048
torch                          mean time: 0.016528 ms
cumsum_fp32                    mean time: 0.011456 ms, speedup: 1.44
cumsum_fp32x4                  mean time: 0.008864 ms, speedup: 1.86
####################################################################################################
n: 32, m: 4096
torch                          mean time: 0.017520 ms
cumsum_fp32                    mean time: 0.017616 ms, speedup: 0.99
cumsum_fp32x4                  mean time: 0.011104 ms, speedup: 1.58
####################################################################################################
n: 32, m: 8192
torch                          mean time: 0.028960 ms
cumsum_fp32                    mean time: 0.027744 ms, speedup: 1.04
cumsum_fp32x4                  mean time: 0.016640 ms, speedup: 1.74
####################################################################################################
n: 32, m: 12800
torch                          mean time: 0.029040 ms
cumsum_fp32                    mean time: 0.041808 ms, speedup: 0.69
cumsum_fp32x4                  mean time: 0.021488 ms, speedup: 1.35
####################################################################################################
n: 64, m: 2048
torch                          mean time: 0.017152 ms
cumsum_fp32                    mean time: 0.013040 ms, speedup: 1.32
cumsum_fp32x4                  mean time: 0.009600 ms, speedup: 1.79
####################################################################################################
n: 64, m: 4096
torch                          mean time: 0.027584 ms
cumsum_fp32                    mean time: 0.019392 ms, speedup: 1.42
cumsum_fp32x4                  mean time: 0.012800 ms, speedup: 2.16
####################################################################################################
n: 64, m: 8192
torch                          mean time: 0.033424 ms
cumsum_fp32                    mean time: 0.031712 ms, speedup: 1.05
cumsum_fp32x4                  mean time: 0.018272 ms, speedup: 1.83
####################################################################################################
n: 64, m: 12800
torch                          mean time: 0.048256 ms
cumsum_fp32                    mean time: 0.047424 ms, speedup: 1.02
cumsum_fp32x4                  mean time: 0.024960 ms, speedup: 1.93
####################################################################################################
n: 128, m: 2048
torch                          mean time: 0.025712 ms
cumsum_fp32                    mean time: 0.015360 ms, speedup: 1.67
cumsum_fp32x4                  mean time: 0.010688 ms, speedup: 2.41
####################################################################################################
n: 128, m: 4096
torch                          mean time: 0.031552 ms
cumsum_fp32                    mean time: 0.025024 ms, speedup: 1.26
cumsum_fp32x4                  mean time: 0.013408 ms, speedup: 2.35
####################################################################################################
n: 128, m: 8192
torch                          mean time: 0.056864 ms
cumsum_fp32                    mean time: 0.042976 ms, speedup: 1.32
cumsum_fp32x4                  mean time: 0.025696 ms, speedup: 2.21
####################################################################################################
n: 128, m: 12800
torch                          mean time: 0.062368 ms
cumsum_fp32                    mean time: 0.060608 ms, speedup: 1.03
cumsum_fp32x4                  mean time: 0.049120 ms, speedup: 1.27
```
