# rmsnorm

## Overview

RMSNorm kernels.

- [x] naive Torch RMSNorm
- [x] rmsnorm — FP32 / FP16
- [x] rmsnorm_fp32x4 — vectorized FP32
- [x] rmsnorm_fp32x4_smem
- [x] rmsnorm_fp16x8 — vectorized FP16, packed r/w
- [x] rmsnorm_fp16x8_smem — vectorized FP16, packed r/w
- [x] pytorch op bindings && diff check

## Run tests

L2 is large enough that staging weights in shared memory barely helps; the bottleneck stays input/output traffic.

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### Sample output

```bash
####################################################################################################
n: 256, m: 2048
torch                          mean time: 0.048896 ms
rmsnorm                        mean time: 0.017904 ms, speedup: 2.73
rmsnorm_fp32x4                 mean time: 0.017024 ms, speedup: 2.87
rmsnorm_fp32x4_smem            mean time: 0.035392 ms, speedup: 1.38
torch                          mean time: 0.043360 ms
rmsnorm_fp16                   mean time: 0.013856 ms, speedup: 3.13
rmsnorm_fp16x8_packed          mean time: 0.011712 ms, speedup: 3.70
rmsnorm_fp16x8_packed_smem     mean time: 0.022912 ms, speedup: 1.89
####################################################################################################
n: 256, m: 4096
torch                          mean time: 0.068176 ms
rmsnorm                        mean time: 0.037872 ms, speedup: 1.80
rmsnorm_fp32x4                 mean time: 0.035664 ms, speedup: 1.91
rmsnorm_fp32x4_smem            mean time: 0.048528 ms, speedup: 1.40
torch                          mean time: 0.054528 ms
rmsnorm_fp16                   mean time: 0.027488 ms, speedup: 1.98
rmsnorm_fp16x8_packed          mean time: 0.015488 ms, speedup: 3.52
rmsnorm_fp16x8_packed_smem     mean time: 0.035136 ms, speedup: 1.55
####################################################################################################
n: 256, m: 8192
torch                          mean time: 0.111424 ms
rmsnorm                        mean time: 0.070144 ms, speedup: 1.59
rmsnorm_fp32x4                 mean time: 0.066736 ms, speedup: 1.67
rmsnorm_fp32x4_smem            mean time: 0.084000 ms, speedup: 1.33
torch                          mean time: 0.082208 ms
rmsnorm_fp16                   mean time: 0.051776 ms, speedup: 1.59
rmsnorm_fp16x8_packed          mean time: 0.035312 ms, speedup: 2.33
rmsnorm_fp16x8_packed_smem     mean time: 0.058320 ms, speedup: 1.41
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.110224 ms
rmsnorm                        mean time: 0.057296 ms, speedup: 1.92
rmsnorm_fp32x4                 mean time: 0.055280 ms, speedup: 1.99
rmsnorm_fp32x4_smem            mean time: 0.062512 ms, speedup: 1.76
torch                          mean time: 0.080224 ms
rmsnorm_fp16                   mean time: 0.038096 ms, speedup: 2.11
rmsnorm_fp16x8_packed          mean time: 0.029408 ms, speedup: 2.73
rmsnorm_fp16x8_packed_smem     mean time: 0.030656 ms, speedup: 2.62
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.225952 ms
rmsnorm                        mean time: 0.110352 ms, speedup: 2.05
rmsnorm_fp32x4                 mean time: 0.110480 ms, speedup: 2.05
rmsnorm_fp32x4_smem            mean time: 0.121920 ms, speedup: 1.85
torch                          mean time: 0.143536 ms
rmsnorm_fp16                   mean time: 0.067456 ms, speedup: 2.13
rmsnorm_fp16x8_packed          mean time: 0.055776 ms, speedup: 2.57
rmsnorm_fp16x8_packed_smem     mean time: 0.061840 ms, speedup: 2.32
####################################################################################################
n: 1024, m: 8192
torch                          mean time: 0.690928 ms
rmsnorm                        mean time: 0.225120 ms, speedup: 3.07
rmsnorm_fp32x4                 mean time: 0.221680 ms, speedup: 3.12
rmsnorm_fp32x4_smem            mean time: 0.235568 ms, speedup: 2.93
torch                          mean time: 0.322640 ms
rmsnorm_fp16                   mean time: 0.130928 ms, speedup: 2.46
rmsnorm_fp16x8_packed          mean time: 0.109712 ms, speedup: 2.94
rmsnorm_fp16x8_packed_smem     mean time: 0.120640 ms, speedup: 2.67
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.228112 ms
rmsnorm                        mean time: 0.107552 ms, speedup: 2.12
rmsnorm_fp32x4                 mean time: 0.140656 ms, speedup: 1.62
rmsnorm_fp32x4_smem            mean time: 0.146576 ms, speedup: 1.56
torch                          mean time: 0.145328 ms
rmsnorm_fp16                   mean time: 0.061856 ms, speedup: 2.35
rmsnorm_fp16x8_packed          mean time: 0.054608 ms, speedup: 2.66
rmsnorm_fp16x8_packed_smem     mean time: 0.060128 ms, speedup: 2.42
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.681024 ms
rmsnorm                        mean time: 0.216464 ms, speedup: 3.15
rmsnorm_fp32x4                 mean time: 0.214032 ms, speedup: 3.18
rmsnorm_fp32x4_smem            mean time: 0.234416 ms, speedup: 2.91
torch                          mean time: 0.310768 ms
rmsnorm_fp16                   mean time: 0.119808 ms, speedup: 2.59
rmsnorm_fp16x8_packed          mean time: 0.105840 ms, speedup: 2.94
rmsnorm_fp16x8_packed_smem     mean time: 0.112368 ms, speedup: 2.77
####################################################################################################
n: 2048, m: 8192
torch                          mean time: 1.384640 ms
rmsnorm                        mean time: 0.427536 ms, speedup: 3.24
rmsnorm_fp32x4                 mean time: 0.422592 ms, speedup: 3.28
rmsnorm_fp32x4_smem            mean time: 0.432512 ms, speedup: 3.20
torch                          mean time: 0.810768 ms
rmsnorm_fp16                   mean time: 0.231808 ms, speedup: 3.50
rmsnorm_fp16x8_packed          mean time: 0.214608 ms, speedup: 3.78
rmsnorm_fp16x8_packed_smem     mean time: 0.227728 ms, speedup: 3.56
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.682512 ms
rmsnorm                        mean time: 0.211136 ms, speedup: 3.23
rmsnorm_fp32x4                 mean time: 0.210784 ms, speedup: 3.24
rmsnorm_fp32x4_smem            mean time: 0.225840 ms, speedup: 3.02
torch                          mean time: 0.304416 ms
rmsnorm_fp16                   mean time: 0.113632 ms, speedup: 2.68
rmsnorm_fp16x8_packed          mean time: 0.103696 ms, speedup: 2.94
rmsnorm_fp16x8_packed_smem     mean time: 0.124896 ms, speedup: 2.44
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 1.392272 ms
rmsnorm                        mean time: 0.418160 ms, speedup: 3.33
rmsnorm_fp32x4                 mean time: 0.416608 ms, speedup: 3.34
rmsnorm_fp32x4_smem            mean time: 0.439072 ms, speedup: 3.17
torch                          mean time: 0.886032 ms
rmsnorm_fp16                   mean time: 0.226544 ms, speedup: 3.91
rmsnorm_fp16x8_packed          mean time: 0.213680 ms, speedup: 4.15
rmsnorm_fp16x8_packed_smem     mean time: 0.228512 ms, speedup: 3.88
####################################################################################################
n: 4096, m: 8192
torch                          mean time: 2.763136 ms
rmsnorm                        mean time: 0.833040 ms, speedup: 3.32
rmsnorm_fp32x4                 mean time: 0.825648 ms, speedup: 3.35
rmsnorm_fp32x4_smem            mean time: 0.838336 ms, speedup: 3.30
torch                          mean time: 2.015728 ms
rmsnorm_fp16                   mean time: 0.466784 ms, speedup: 4.32
rmsnorm_fp16x8_packed          mean time: 0.415360 ms, speedup: 4.85
rmsnorm_fp16x8_packed_smem     mean time: 0.435680 ms, speedup: 4.63
```
