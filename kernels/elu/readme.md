# Elu

## Overview

ELU kernels.

- [x] elu — FP32 / FP16
- [x] elu_fp16x2 — vectorized FP16
- [x] elu_fp16x8 — vectorized FP16
- [x] elu_fp16x8_packed — vectorized FP16, packed r/w (~2× via `half2`)
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
torch                          mean time: 0.028800 ms
elu                            mean time: 0.035456 ms, speedup: 0.81
elu_fp32x4                     mean time: 0.025216 ms, speedup: 1.14
torch                          mean time: 0.014704 ms
elu_half                       mean time: 0.028656 ms, speedup: 0.51
elu_fp16x2                     mean time: 0.019488 ms, speedup: 0.75
elu_fp16x8                     mean time: 0.016880 ms, speedup: 0.87
elu_fp16x8_packed              mean time: 0.013120 ms, speedup: 1.12
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.049184 ms
elu                            mean time: 0.060048 ms, speedup: 0.82
elu_fp32x4                     mean time: 0.051952 ms, speedup: 0.95
torch                          mean time: 0.023264 ms
elu_half                       mean time: 0.050608 ms, speedup: 0.46
elu_fp16x2                     mean time: 0.033680 ms, speedup: 0.69
elu_fp16x8                     mean time: 0.031120 ms, speedup: 0.75
elu_fp16x8_packed              mean time: 0.023584 ms, speedup: 0.99
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.101216 ms
elu                            mean time: 0.114704 ms, speedup: 0.88
elu_fp32x4                     mean time: 0.135376 ms, speedup: 0.75
torch                          mean time: 0.060976 ms
elu_half                       mean time: 0.091312 ms, speedup: 0.67
elu_fp16x2                     mean time: 0.062320 ms, speedup: 0.98
elu_fp16x8                     mean time: 0.060640 ms, speedup: 1.01
elu_fp16x8_packed              mean time: 0.053696 ms, speedup: 1.14
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.049936 ms
elu                            mean time: 0.060656 ms, speedup: 0.82
elu_fp32x4                     mean time: 0.051344 ms, speedup: 0.97
torch                          mean time: 0.023744 ms
elu_half                       mean time: 0.054720 ms, speedup: 0.43
elu_fp16x2                     mean time: 0.037632 ms, speedup: 0.63
elu_fp16x8                     mean time: 0.034416 ms, speedup: 0.69
elu_fp16x8_packed              mean time: 0.023360 ms, speedup: 1.02
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.102304 ms
elu                            mean time: 0.132048 ms, speedup: 0.77
elu_fp32x4                     mean time: 0.106192 ms, speedup: 0.96
torch                          mean time: 0.047200 ms
elu_half                       mean time: 0.095584 ms, speedup: 0.49
elu_fp16x2                     mean time: 0.062480 ms, speedup: 0.76
elu_fp16x8                     mean time: 0.059296 ms, speedup: 0.80
elu_fp16x8_packed              mean time: 0.052192 ms, speedup: 0.90
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.204416 ms
elu                            mean time: 0.228704 ms, speedup: 0.89
elu_fp32x4                     mean time: 0.204416 ms, speedup: 1.00
torch                          mean time: 0.102416 ms
elu_half                       mean time: 0.171552 ms, speedup: 0.60
elu_fp16x2                     mean time: 0.116032 ms, speedup: 0.88
elu_fp16x8                     mean time: 0.109552 ms, speedup: 0.93
elu_fp16x8_packed              mean time: 0.103024 ms, speedup: 0.99
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.101312 ms
elu                            mean time: 0.116368 ms, speedup: 0.87
elu_fp32x4                     mean time: 0.102880 ms, speedup: 0.98
torch                          mean time: 0.046768 ms
elu_half                       mean time: 0.093760 ms, speedup: 0.50
elu_fp16x2                     mean time: 0.062336 ms, speedup: 0.75
elu_fp16x8                     mean time: 0.060640 ms, speedup: 0.77
elu_fp16x8_packed              mean time: 0.052688 ms, speedup: 0.89
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.205088 ms
elu                            mean time: 0.233424 ms, speedup: 0.88
elu_fp32x4                     mean time: 0.207152 ms, speedup: 0.99
torch                          mean time: 0.102016 ms
elu_half                       mean time: 0.183840 ms, speedup: 0.55
elu_fp16x2                     mean time: 0.121744 ms, speedup: 0.84
elu_fp16x8                     mean time: 0.111776 ms, speedup: 0.91
elu_fp16x8_packed              mean time: 0.104096 ms, speedup: 0.98
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.403728 ms
elu                            mean time: 0.438784 ms, speedup: 0.92
elu_fp32x4                     mean time: 0.426912 ms, speedup: 0.95
torch                          mean time: 0.203392 ms
elu_half                       mean time: 0.325744 ms, speedup: 0.62
elu_fp16x2                     mean time: 0.227152 ms, speedup: 0.90
elu_fp16x8                     mean time: 0.207440 ms, speedup: 0.98
elu_fp16x8_packed              mean time: 0.204368 ms, speedup: 1.00
```
