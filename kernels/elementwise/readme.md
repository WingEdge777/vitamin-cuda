# Elementwise

## Overview

Element-wise addition kernels.

- [x] elementwise_add — FP32 / FP16
- [x] elementwise_add_fp16x2 — vectorized FP16
- [x] elementwise_add_fp16x8 — vectorized FP16
- [x] elementwise_add_fp16x8_packed — vectorized FP16, packed r/w
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
torch                          mean time: 0.047024 ms
elementwise_add                mean time: 0.042432 ms, speedup: 1.11
elementwise_add_fp32x4         mean time: 0.046096 ms, speedup: 1.02
torch                          mean time: 0.023920 ms
ele_add_half                   mean time: 0.035584 ms, speedup: 0.67
elementwise_add_fp16x2         mean time: 0.021312 ms, speedup: 1.12
elementwise_add_fp16x8         mean time: 0.023264 ms, speedup: 1.03
elementwise_add_fp16x8_packed  mean time: 0.022320 ms, speedup: 1.07
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.077600 ms
elementwise_add                mean time: 0.079104 ms, speedup: 0.98
elementwise_add_fp32x4         mean time: 0.078528 ms, speedup: 0.99
torch                          mean time: 0.042800 ms
ele_add_half                   mean time: 0.064288 ms, speedup: 0.67
elementwise_add_fp16x2         mean time: 0.040480 ms, speedup: 1.06
elementwise_add_fp16x8         mean time: 0.043008 ms, speedup: 1.00
elementwise_add_fp16x8_packed  mean time: 0.042512 ms, speedup: 1.01
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.156848 ms
elementwise_add                mean time: 0.160432 ms, speedup: 0.98
elementwise_add_fp32x4         mean time: 0.155312 ms, speedup: 1.01
torch                          mean time: 0.080832 ms
ele_add_half                   mean time: 0.107856 ms, speedup: 0.75
elementwise_add_fp16x2         mean time: 0.078000 ms, speedup: 1.04
elementwise_add_fp16x8         mean time: 0.079376 ms, speedup: 1.02
elementwise_add_fp16x8_packed  mean time: 0.077968 ms, speedup: 1.04
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.079072 ms
elementwise_add                mean time: 0.078576 ms, speedup: 1.01
elementwise_add_fp32x4         mean time: 0.077648 ms, speedup: 1.02
torch                          mean time: 0.039728 ms
ele_add_half                   mean time: 0.061088 ms, speedup: 0.65
elementwise_add_fp16x2         mean time: 0.038048 ms, speedup: 1.04
elementwise_add_fp16x8         mean time: 0.041312 ms, speedup: 0.96
elementwise_add_fp16x8_packed  mean time: 0.039024 ms, speedup: 1.02
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.154752 ms
elementwise_add                mean time: 0.161184 ms, speedup: 0.96
elementwise_add_fp32x4         mean time: 0.156640 ms, speedup: 0.99
torch                          mean time: 0.079744 ms
ele_add_half                   mean time: 0.106752 ms, speedup: 0.75
elementwise_add_fp16x2         mean time: 0.076768 ms, speedup: 1.04
elementwise_add_fp16x8         mean time: 0.079872 ms, speedup: 1.00
elementwise_add_fp16x8_packed  mean time: 0.078480 ms, speedup: 1.02
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.302624 ms
elementwise_add                mean time: 0.305248 ms, speedup: 0.99
elementwise_add_fp32x4         mean time: 0.302080 ms, speedup: 1.00
torch                          mean time: 0.156864 ms
ele_add_half                   mean time: 0.197952 ms, speedup: 0.79
elementwise_add_fp16x2         mean time: 0.160304 ms, speedup: 0.98
elementwise_add_fp16x8         mean time: 0.157040 ms, speedup: 1.00
elementwise_add_fp16x8_packed  mean time: 0.157568 ms, speedup: 1.00
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.153680 ms
elementwise_add                mean time: 0.159424 ms, speedup: 0.96
elementwise_add_fp32x4         mean time: 0.156560 ms, speedup: 0.98
torch                          mean time: 0.080432 ms
ele_add_half                   mean time: 0.107808 ms, speedup: 0.75
elementwise_add_fp16x2         mean time: 0.078832 ms, speedup: 1.02
elementwise_add_fp16x8         mean time: 0.079408 ms, speedup: 1.01
elementwise_add_fp16x8_packed  mean time: 0.077584 ms, speedup: 1.04
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.303888 ms
elementwise_add                mean time: 0.307264 ms, speedup: 0.99
elementwise_add_fp32x4         mean time: 0.302288 ms, speedup: 1.01
torch                          mean time: 0.154736 ms
ele_add_half                   mean time: 0.195632 ms, speedup: 0.79
elementwise_add_fp16x2         mean time: 0.158896 ms, speedup: 0.97
elementwise_add_fp16x8         mean time: 0.156992 ms, speedup: 0.99
elementwise_add_fp16x8_packed  mean time: 0.156432 ms, speedup: 0.99
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.595072 ms
elementwise_add                mean time: 0.594800 ms, speedup: 1.00
elementwise_add_fp32x4         mean time: 0.589728 ms, speedup: 1.01
torch                          mean time: 0.303232 ms
ele_add_half                   mean time: 0.357264 ms, speedup: 0.85
elementwise_add_fp16x2         mean time: 0.306688 ms, speedup: 0.99
elementwise_add_fp16x8         mean time: 0.306144 ms, speedup: 0.99
elementwise_add_fp16x8_packed  mean time: 0.305952 ms, speedup: 0.99
```
