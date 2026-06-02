# Relu

## Overview

ReLU kernels.

- [x] relu — FP32 / FP16
- [x] relu_fp16x2 — vectorized FP16
- [x] relu_fp16x8 — vectorized FP16
- [x] relu_fp16x8_packed — vectorized FP16, packed r/w
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
torch                          mean time: 0.028208 ms
relu                           mean time: 0.036064 ms, speedup: 0.78
relu_fp32x4                    mean time: 0.025104 ms, speedup: 1.12
torch                          mean time: 0.015424 ms
relu_half                      mean time: 0.027632 ms, speedup: 0.56
relu_fp16x2                    mean time: 0.019264 ms, speedup: 0.80
relu_fp16x8                    mean time: 0.013136 ms, speedup: 1.17
relu_fp16x8_packed             mean time: 0.013232 ms, speedup: 1.17
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.050480 ms
relu                           mean time: 0.060208 ms, speedup: 0.84
relu_fp32x4                    mean time: 0.052480 ms, speedup: 0.96
torch                          mean time: 0.022688 ms
relu_half                      mean time: 0.048416 ms, speedup: 0.47
relu_fp16x2                    mean time: 0.033328 ms, speedup: 0.68
relu_fp16x8                    mean time: 0.023728 ms, speedup: 0.96
relu_fp16x8_packed             mean time: 0.023472 ms, speedup: 0.97
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.101984 ms
relu                           mean time: 0.114736 ms, speedup: 0.89
relu_fp32x4                    mean time: 0.102752 ms, speedup: 0.99
torch                          mean time: 0.049808 ms
relu_half                      mean time: 0.086016 ms, speedup: 0.58
relu_fp16x2                    mean time: 0.068144 ms, speedup: 0.73
relu_fp16x8                    mean time: 0.052464 ms, speedup: 0.95
relu_fp16x8_packed             mean time: 0.053008 ms, speedup: 0.94
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.050016 ms
relu                           mean time: 0.061392 ms, speedup: 0.81
relu_fp32x4                    mean time: 0.050944 ms, speedup: 0.98
torch                          mean time: 0.023600 ms
relu_half                      mean time: 0.048320 ms, speedup: 0.49
relu_fp16x2                    mean time: 0.033376 ms, speedup: 0.71
relu_fp16x8                    mean time: 0.023360 ms, speedup: 1.01
relu_fp16x8_packed             mean time: 0.023104 ms, speedup: 1.02
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.102352 ms
relu                           mean time: 0.116256 ms, speedup: 0.88
relu_fp32x4                    mean time: 0.105664 ms, speedup: 0.97
torch                          mean time: 0.052192 ms
relu_half                      mean time: 0.093072 ms, speedup: 0.56
relu_fp16x2                    mean time: 0.061760 ms, speedup: 0.85
relu_fp16x8                    mean time: 0.052064 ms, speedup: 1.00
relu_fp16x8_packed             mean time: 0.051744 ms, speedup: 1.01
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.206976 ms
relu                           mean time: 0.225184 ms, speedup: 0.92
relu_fp32x4                    mean time: 0.204224 ms, speedup: 1.01
torch                          mean time: 0.102464 ms
relu_half                      mean time: 0.161728 ms, speedup: 0.63
relu_fp16x2                    mean time: 0.113680 ms, speedup: 0.90
relu_fp16x8                    mean time: 0.101888 ms, speedup: 1.01
relu_fp16x8_packed             mean time: 0.103152 ms, speedup: 0.99
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.103168 ms
relu                           mean time: 0.114448 ms, speedup: 0.90
relu_fp32x4                    mean time: 0.102240 ms, speedup: 1.01
torch                          mean time: 0.049008 ms
relu_half                      mean time: 0.088944 ms, speedup: 0.55
relu_fp16x2                    mean time: 0.060928 ms, speedup: 0.80
relu_fp16x8                    mean time: 0.052928 ms, speedup: 0.93
relu_fp16x8_packed             mean time: 0.052160 ms, speedup: 0.94
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.204448 ms
relu                           mean time: 0.230944 ms, speedup: 0.89
relu_fp32x4                    mean time: 0.206416 ms, speedup: 0.99
torch                          mean time: 0.102112 ms
relu_half                      mean time: 0.180928 ms, speedup: 0.56
relu_fp16x2                    mean time: 0.120272 ms, speedup: 0.85
relu_fp16x8                    mean time: 0.103776 ms, speedup: 0.98
relu_fp16x8_packed             mean time: 0.103568 ms, speedup: 0.99
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.406832 ms
relu                           mean time: 0.415664 ms, speedup: 0.98
relu_fp32x4                    mean time: 0.401632 ms, speedup: 1.01
torch                          mean time: 0.205040 ms
relu_half                      mean time: 0.304336 ms, speedup: 0.67
relu_fp16x2                    mean time: 0.224896 ms, speedup: 0.91
relu_fp16x8                    mean time: 0.203856 ms, speedup: 1.01
relu_fp16x8_packed             mean time: 0.203728 ms, speedup: 1.01
```
