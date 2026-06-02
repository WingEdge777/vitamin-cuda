# relu6

## Overview

ReLU6 kernels.

- [x] relu6 — FP32 / FP16
- [x] relu6_fp16x2 — vectorized FP16
- [x] relu6_fp16x8 — vectorized FP16
- [x] relu6_fp16x8_packed — vectorized FP16, packed r/w
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
torch                          mean time: 0.029376 ms
relu6                          mean time: 0.037056 ms, speedup: 0.79
relu6_fp32x4                   mean time: 0.026176 ms, speedup: 1.12
torch                          mean time: 0.013648 ms
relu6_half                     mean time: 0.027648 ms, speedup: 0.49
relu6_fp16x2                   mean time: 0.020512 ms, speedup: 0.67
relu6_fp16x8                   mean time: 0.013424 ms, speedup: 1.02
relu6_fp16x8_packed            mean time: 0.013152 ms, speedup: 1.04
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.048800 ms
relu6                          mean time: 0.060016 ms, speedup: 0.81
relu6_fp32x4                   mean time: 0.052432 ms, speedup: 0.93
torch                          mean time: 0.023440 ms
relu6_half                     mean time: 0.048608 ms, speedup: 0.48
relu6_fp16x2                   mean time: 0.033744 ms, speedup: 0.69
relu6_fp16x8                   mean time: 0.023696 ms, speedup: 0.99
relu6_fp16x8_packed            mean time: 0.024240 ms, speedup: 0.97
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.101152 ms
relu6                          mean time: 0.114240 ms, speedup: 0.89
relu6_fp32x4                   mean time: 0.103472 ms, speedup: 0.98
torch                          mean time: 0.046288 ms
relu6_half                     mean time: 0.086912 ms, speedup: 0.53
relu6_fp16x2                   mean time: 0.060560 ms, speedup: 0.76
relu6_fp16x8                   mean time: 0.053456 ms, speedup: 0.87
relu6_fp16x8_packed            mean time: 0.053696 ms, speedup: 0.86
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.049040 ms
relu6                          mean time: 0.060208 ms, speedup: 0.81
relu6_fp32x4                   mean time: 0.050560 ms, speedup: 0.97
torch                          mean time: 0.021792 ms
relu6_half                     mean time: 0.048416 ms, speedup: 0.45
relu6_fp16x2                   mean time: 0.033392 ms, speedup: 0.65
relu6_fp16x8                   mean time: 0.024480 ms, speedup: 0.89
relu6_fp16x8_packed            mean time: 0.023392 ms, speedup: 0.93
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.100592 ms
relu6                          mean time: 0.114544 ms, speedup: 0.88
relu6_fp32x4                   mean time: 0.105776 ms, speedup: 0.95
torch                          mean time: 0.046288 ms
relu6_half                     mean time: 0.086832 ms, speedup: 0.53
relu6_fp16x2                   mean time: 0.060544 ms, speedup: 0.76
relu6_fp16x8                   mean time: 0.052048 ms, speedup: 0.89
relu6_fp16x8_packed            mean time: 0.052272 ms, speedup: 0.89
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.202784 ms
relu6                          mean time: 0.225776 ms, speedup: 0.90
relu6_fp32x4                   mean time: 0.204032 ms, speedup: 0.99
torch                          mean time: 0.100800 ms
relu6_half                     mean time: 0.160720 ms, speedup: 0.63
relu6_fp16x2                   mean time: 0.113088 ms, speedup: 0.89
relu6_fp16x8                   mean time: 0.103248 ms, speedup: 0.98
relu6_fp16x8_packed            mean time: 0.102864 ms, speedup: 0.98
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.100272 ms
relu6                          mean time: 0.114928 ms, speedup: 0.87
relu6_fp32x4                   mean time: 0.103392 ms, speedup: 0.97
torch                          mean time: 0.046000 ms
relu6_half                     mean time: 0.088832 ms, speedup: 0.52
relu6_fp16x2                   mean time: 0.060416 ms, speedup: 0.76
relu6_fp16x8                   mean time: 0.052560 ms, speedup: 0.88
relu6_fp16x8_packed            mean time: 0.052544 ms, speedup: 0.88
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.202960 ms
relu6                          mean time: 0.225776 ms, speedup: 0.90
relu6_fp32x4                   mean time: 0.205088 ms, speedup: 0.99
torch                          mean time: 0.102128 ms
relu6_half                     mean time: 0.162352 ms, speedup: 0.63
relu6_fp16x2                   mean time: 0.113600 ms, speedup: 0.90
relu6_fp16x8                   mean time: 0.103840 ms, speedup: 0.98
relu6_fp16x8_packed            mean time: 0.103616 ms, speedup: 0.99
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.401920 ms
relu6                          mean time: 0.414784 ms, speedup: 0.97
relu6_fp32x4                   mean time: 0.400976 ms, speedup: 1.00
torch                          mean time: 0.258784 ms
relu6_half                     mean time: 0.356512 ms, speedup: 0.73
relu6_fp16x2                   mean time: 0.268176 ms, speedup: 0.96
relu6_fp16x8                   mean time: 0.204752 ms, speedup: 1.26
relu6_fp16x8_packed            mean time: 0.204704 ms, speedup: 1.26
```
