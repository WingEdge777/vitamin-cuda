# Rope

## Overview

RoPE kernels.

- [x] PyTorch naive RoPE
- [x] PyTorch RoPE with cos/sin cache
- [x] rope — FP32 (~10× vs naive PyTorch)
- [x] rope_fp32x4 — vectorized FP32 (much faster still)
- [x] pytorch op bindings && diff check

## Run tests

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### Sample output

```bash
[2/2] c++ rope_neox.cuda.o -shared -L/home/zhu/.venv/lib/python3.12/site-packages/torch/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch -ltorch_python -L/usr/local/cuda/lib64 -lcudart -o rope_lib.so
####################################################################################################
bs: 64, n: 512, m: 128
torch                          mean time: 0.629776 ms
torch.rope_with_sin_cos_cache  mean time: 0.403872 ms, speedup: 1.56
rope                           mean time: 0.104128 ms, speedup: 6.05
rope_fp32x4                    mean time: 0.103712 ms, speedup: 6.07
####################################################################################################
bs: 64, n: 512, m: 256
torch                          mean time: 1.328352 ms
torch.rope_with_sin_cos_cache  mean time: 1.302448 ms, speedup: 1.02
rope                           mean time: 0.264672 ms, speedup: 5.02
rope_fp32x4                    mean time: 0.219296 ms, speedup: 6.06
####################################################################################################
bs: 64, n: 1024, m: 128
torch                          mean time: 1.380832 ms
torch.rope_with_sin_cos_cache  mean time: 1.112848 ms, speedup: 1.24
rope                           mean time: 0.220544 ms, speedup: 6.26
rope_fp32x4                    mean time: 0.222400 ms, speedup: 6.21
####################################################################################################
bs: 64, n: 1024, m: 256
torch                          mean time: 2.375216 ms
torch.rope_with_sin_cos_cache  mean time: 2.116256 ms, speedup: 1.12
rope                           mean time: 0.406464 ms, speedup: 5.84
rope_fp32x4                    mean time: 0.406032 ms, speedup: 5.85
####################################################################################################
bs: 64, n: 2048, m: 128
torch                          mean time: 5.923248 ms
torch.rope_with_sin_cos_cache  mean time: 9.245568 ms, speedup: 0.64
rope                           mean time: 1.759664 ms, speedup: 3.37
rope_fp32x4                    mean time: 1.052368 ms, speedup: 5.63
####################################################################################################
bs: 64, n: 2048, m: 256
torch                          mean time: 19.124656 ms
torch.rope_with_sin_cos_cache  mean time: 18.360175 ms, speedup: 1.04
rope                           mean time: 2.281120 ms, speedup: 8.38
rope_fp32x4                    mean time: 2.097632 ms, speedup: 9.12
####################################################################################################
bs: 64, n: 4096, m: 128
torch                          mean time: 18.851536 ms
torch.rope_with_sin_cos_cache  mean time: 8.220720 ms, speedup: 2.29
rope                           mean time: 1.326960 ms, speedup: 14.21
rope_fp32x4                    mean time: 1.034064 ms, speedup: 18.23
####################################################################################################
bs: 64, n: 4096, m: 256
torch                          mean time: 11.435872 ms
torch.rope_with_sin_cos_cache  mean time: 8.531152 ms, speedup: 1.34
rope                           mean time: 1.593696 ms, speedup: 7.18
rope_fp32x4                    mean time: 1.602096 ms, speedup: 7.14
####################################################################################################
bs: 64, n: 8192, m: 128
torch                          mean time: 9.198768 ms
torch.rope_with_sin_cos_cache  mean time: 8.668448 ms, speedup: 1.06
rope                           mean time: 1.608800 ms, speedup: 5.72
rope_fp32x4                    mean time: 1.599552 ms, speedup: 5.75
####################################################################################################
bs: 64, n: 8192, m: 256
torch                          mean time: 26.261985 ms
torch.rope_with_sin_cos_cache  mean time: 21.048672 ms, speedup: 1.25
rope                           mean time: 3.372080 ms, speedup: 7.79
rope_fp32x4                    mean time: 3.209040 ms, speedup: 8.18
####################################################################################################
bs: 128, n: 512, m: 128
torch                          mean time: 1.220208 ms
torch.rope_with_sin_cos_cache  mean time: 1.042320 ms, speedup: 1.17
rope                           mean time: 0.208048 ms, speedup: 5.87
rope_fp32x4                    mean time: 0.206576 ms, speedup: 5.91
####################################################################################################
bs: 128, n: 512, m: 256
torch                          mean time: 2.430912 ms
torch.rope_with_sin_cos_cache  mean time: 2.208752 ms, speedup: 1.10
rope                           mean time: 0.407376 ms, speedup: 5.97
rope_fp32x4                    mean time: 0.405184 ms, speedup: 6.00
####################################################################################################
bs: 128, n: 1024, m: 128
torch                          mean time: 2.410448 ms
torch.rope_with_sin_cos_cache  mean time: 2.157168 ms, speedup: 1.12
rope                           mean time: 0.405920 ms, speedup: 5.94
rope_fp32x4                    mean time: 0.407440 ms, speedup: 5.92
####################################################################################################
bs: 128, n: 1024, m: 256
torch                          mean time: 4.528208 ms
torch.rope_with_sin_cos_cache  mean time: 4.607776 ms, speedup: 0.98
rope                           mean time: 0.850688 ms, speedup: 5.32
rope_fp32x4                    mean time: 0.808400 ms, speedup: 5.60
####################################################################################################
bs: 128, n: 2048, m: 128
torch                          mean time: 4.607456 ms
torch.rope_with_sin_cos_cache  mean time: 4.312800 ms, speedup: 1.07
rope                           mean time: 0.802944 ms, speedup: 5.74
rope_fp32x4                    mean time: 0.803680 ms, speedup: 5.73
####################################################################################################
bs: 128, n: 2048, m: 256
torch                          mean time: 9.205872 ms
torch.rope_with_sin_cos_cache  mean time: 8.598848 ms, speedup: 1.07
rope                           mean time: 1.596368 ms, speedup: 5.77
rope_fp32x4                    mean time: 1.602224 ms, speedup: 5.75
####################################################################################################
bs: 128, n: 4096, m: 128
torch                          mean time: 9.226656 ms
torch.rope_with_sin_cos_cache  mean time: 9.239072 ms, speedup: 1.00
rope                           mean time: 1.609360 ms, speedup: 5.73
rope_fp32x4                    mean time: 1.600928 ms, speedup: 5.76
####################################################################################################
bs: 128, n: 4096, m: 256
torch                          mean time: 18.362976 ms
torch.rope_with_sin_cos_cache  mean time: 17.954513 ms, speedup: 1.02
rope                           mean time: 3.183872 ms, speedup: 5.77
rope_fp32x4                    mean time: 3.200208 ms, speedup: 5.74
####################################################################################################
bs: 128, n: 8192, m: 128
torch                          mean time: 18.864720 ms
torch.rope_with_sin_cos_cache  mean time: 19.260431 ms, speedup: 0.98
rope                           mean time: 3.377216 ms, speedup: 5.59
rope_fp32x4                    mean time: 3.226720 ms, speedup: 5.85
####################################################################################################
bs: 128, n: 8192, m: 256
torch                          mean time: 1000.857086 ms
torch.rope_with_sin_cos_cache  mean time: 978.314728 ms, speedup: 1.02
rope                           mean time: 173.799309 ms, speedup: 5.76
rope_fp32x4                    mean time: 277.359360 ms, speedup: 3.61
```
