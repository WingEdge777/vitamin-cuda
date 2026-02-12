# transpose

## 说明

transpose kernel

- [x] transpose_coalesced_read (input视角，合并读)
- [x] transpose_coalesced_write (output视角，合并写)
- [x] transpose_smem (共享内存缓存，块状读写)
- [x] transpose_smem_bcf (共享内存无冲突版)
- [x] transpose_smem_bcf_packed (共享内存无冲突版，float4向量化读写)
- [x] pytorch op bindings && diff check

## 测试

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### 输出

```bash
####################################################################################################
n: 512, m: 512
torch                          mean time: 0.043646 ms
transpose_coalesced_read       mean time: 0.014394 ms, speedup: 3.03
transpose_coalesced_write      mean time: 0.018390 ms, speedup: 2.37
transpose_smem                 mean time: 0.023125 ms, speedup: 1.89
transpose_smem_bcf             mean time: 0.013589 ms, speedup: 3.21
transpose_smem_bcf_packed      mean time: 0.007966 ms, speedup: 5.48
####################################################################################################
n: 512, m: 1024
torch                          mean time: 0.026097 ms
transpose_coalesced_read       mean time: 0.016641 ms, speedup: 1.57
transpose_coalesced_write      mean time: 0.011788 ms, speedup: 2.21
transpose_smem                 mean time: 0.012481 ms, speedup: 2.09
transpose_smem_bcf             mean time: 0.009518 ms, speedup: 2.74
transpose_smem_bcf_packed      mean time: 0.009199 ms, speedup: 2.84
####################################################################################################
n: 512, m: 2048
torch                          mean time: 0.049567 ms
transpose_coalesced_read       mean time: 0.023081 ms, speedup: 2.15
transpose_coalesced_write      mean time: 0.014432 ms, speedup: 3.43
transpose_smem                 mean time: 0.021489 ms, speedup: 2.31
transpose_smem_bcf             mean time: 0.008807 ms, speedup: 5.63
transpose_smem_bcf_packed      mean time: 0.009777 ms, speedup: 5.07
####################################################################################################
n: 512, m: 4096
torch                          mean time: 0.093000 ms
transpose_coalesced_read       mean time: 0.036096 ms, speedup: 2.58
transpose_coalesced_write      mean time: 0.027273 ms, speedup: 3.41
transpose_smem                 mean time: 0.049260 ms, speedup: 1.89
transpose_smem_bcf             mean time: 0.022123 ms, speedup: 4.20
transpose_smem_bcf_packed      mean time: 0.014786 ms, speedup: 6.29
####################################################################################################
n: 512, m: 8192
torch                          mean time: 0.236194 ms
transpose_coalesced_read       mean time: 0.060854 ms, speedup: 3.88
transpose_coalesced_write      mean time: 0.065378 ms, speedup: 3.61
transpose_smem                 mean time: 0.102891 ms, speedup: 2.30
transpose_smem_bcf             mean time: 0.035771 ms, speedup: 6.60
transpose_smem_bcf_packed      mean time: 0.026798 ms, speedup: 8.81
####################################################################################################
n: 1024, m: 512
torch                          mean time: 0.037596 ms
transpose_coalesced_read       mean time: 0.013671 ms, speedup: 2.75
transpose_coalesced_write      mean time: 0.012081 ms, speedup: 3.11
transpose_smem                 mean time: 0.020409 ms, speedup: 1.84
transpose_smem_bcf             mean time: 0.009874 ms, speedup: 3.81
transpose_smem_bcf_packed      mean time: 0.010988 ms, speedup: 3.42
####################################################################################################
n: 1024, m: 1024
torch                          mean time: 0.071059 ms
transpose_coalesced_read       mean time: 0.021125 ms, speedup: 3.36
transpose_coalesced_write      mean time: 0.015023 ms, speedup: 4.73
transpose_smem                 mean time: 0.029921 ms, speedup: 2.37
transpose_smem_bcf             mean time: 0.011692 ms, speedup: 6.08
transpose_smem_bcf_packed      mean time: 0.011592 ms, speedup: 6.13
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.096413 ms
transpose_coalesced_read       mean time: 0.030398 ms, speedup: 3.17
transpose_coalesced_write      mean time: 0.027566 ms, speedup: 3.50
transpose_smem                 mean time: 0.049794 ms, speedup: 1.94
transpose_smem_bcf             mean time: 0.022012 ms, speedup: 4.38
transpose_smem_bcf_packed      mean time: 0.014511 ms, speedup: 6.64
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.229975 ms
transpose_coalesced_read       mean time: 0.073916 ms, speedup: 3.11
transpose_coalesced_write      mean time: 0.075973 ms, speedup: 3.03
transpose_smem                 mean time: 0.106498 ms, speedup: 2.16
transpose_smem_bcf             mean time: 0.036604 ms, speedup: 6.28
transpose_smem_bcf_packed      mean time: 0.030833 ms, speedup: 7.46
####################################################################################################
n: 1024, m: 8192
torch                          mean time: 0.518723 ms
transpose_coalesced_read       mean time: 0.269519 ms, speedup: 1.92
transpose_coalesced_write      mean time: 0.259618 ms, speedup: 2.00
transpose_smem                 mean time: 0.231551 ms, speedup: 2.24
transpose_smem_bcf             mean time: 0.231331 ms, speedup: 2.24
transpose_smem_bcf_packed      mean time: 0.231509 ms, speedup: 2.24
####################################################################################################
n: 2048, m: 512
torch                          mean time: 0.048923 ms
transpose_coalesced_read       mean time: 0.020123 ms, speedup: 2.43
transpose_coalesced_write      mean time: 0.022714 ms, speedup: 2.15
transpose_smem                 mean time: 0.024387 ms, speedup: 2.01
transpose_smem_bcf             mean time: 0.017194 ms, speedup: 2.85
transpose_smem_bcf_packed      mean time: 0.014143 ms, speedup: 3.46
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.095607 ms
transpose_coalesced_read       mean time: 0.031572 ms, speedup: 3.03
transpose_coalesced_write      mean time: 0.027313 ms, speedup: 3.50
transpose_smem                 mean time: 0.044250 ms, speedup: 2.16
transpose_smem_bcf             mean time: 0.016993 ms, speedup: 5.63
transpose_smem_bcf_packed      mean time: 0.014813 ms, speedup: 6.45
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.238948 ms
transpose_coalesced_read       mean time: 0.080097 ms, speedup: 2.98
transpose_coalesced_write      mean time: 0.082194 ms, speedup: 2.91
transpose_smem                 mean time: 0.133736 ms, speedup: 1.79
transpose_smem_bcf             mean time: 0.031962 ms, speedup: 7.48
transpose_smem_bcf_packed      mean time: 0.029006 ms, speedup: 8.24
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.549302 ms
transpose_coalesced_read       mean time: 0.259774 ms, speedup: 2.11
transpose_coalesced_write      mean time: 0.264194 ms, speedup: 2.08
transpose_smem                 mean time: 0.229236 ms, speedup: 2.40
transpose_smem_bcf             mean time: 0.223347 ms, speedup: 2.46
transpose_smem_bcf_packed      mean time: 0.224166 ms, speedup: 2.45
####################################################################################################
n: 2048, m: 8192
torch                          mean time: 1.368576 ms
transpose_coalesced_read       mean time: 0.539936 ms, speedup: 2.53
transpose_coalesced_write      mean time: 0.508607 ms, speedup: 2.69
transpose_smem                 mean time: 0.556012 ms, speedup: 2.46
transpose_smem_bcf             mean time: 0.445898 ms, speedup: 3.07
transpose_smem_bcf_packed      mean time: 0.444217 ms, speedup: 3.08
####################################################################################################
n: 4096, m: 512
torch                          mean time: 0.093301 ms
transpose_coalesced_read       mean time: 0.042376 ms, speedup: 2.20
transpose_coalesced_write      mean time: 0.038232 ms, speedup: 2.44
transpose_smem                 mean time: 0.053615 ms, speedup: 1.74
transpose_smem_bcf             mean time: 0.021706 ms, speedup: 4.30
transpose_smem_bcf_packed      mean time: 0.014540 ms, speedup: 6.42
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.238892 ms
transpose_coalesced_read       mean time: 0.074521 ms, speedup: 3.21
transpose_coalesced_write      mean time: 0.071440 ms, speedup: 3.34
transpose_smem                 mean time: 0.108631 ms, speedup: 2.20
transpose_smem_bcf             mean time: 0.036070 ms, speedup: 6.62
transpose_smem_bcf_packed      mean time: 0.029182 ms, speedup: 8.19
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.559017 ms
transpose_coalesced_read       mean time: 0.263189 ms, speedup: 2.12
transpose_coalesced_write      mean time: 0.254377 ms, speedup: 2.20
transpose_smem                 mean time: 0.238177 ms, speedup: 2.35
transpose_smem_bcf             mean time: 0.221521 ms, speedup: 2.52
transpose_smem_bcf_packed      mean time: 0.221991 ms, speedup: 2.52
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 1.138517 ms
transpose_coalesced_read       mean time: 0.497994 ms, speedup: 2.29
transpose_coalesced_write      mean time: 0.520331 ms, speedup: 2.19
transpose_smem                 mean time: 0.456160 ms, speedup: 2.50
transpose_smem_bcf             mean time: 0.440512 ms, speedup: 2.58
transpose_smem_bcf_packed      mean time: 0.434925 ms, speedup: 2.62
####################################################################################################
n: 4096, m: 8192
torch                          mean time: 2.235369 ms
transpose_coalesced_read       mean time: 1.092260 ms, speedup: 2.05
transpose_coalesced_write      mean time: 0.983227 ms, speedup: 2.27
transpose_smem                 mean time: 0.845245 ms, speedup: 2.64
transpose_smem_bcf             mean time: 0.869906 ms, speedup: 2.57
transpose_smem_bcf_packed      mean time: 0.868989 ms, speedup: 2.57
####################################################################################################
n: 8192, m: 512
torch                          mean time: 0.241616 ms
transpose_coalesced_read       mean time: 0.065593 ms, speedup: 3.68
transpose_coalesced_write      mean time: 0.066066 ms, speedup: 3.66
transpose_smem                 mean time: 0.096283 ms, speedup: 2.51
transpose_smem_bcf             mean time: 0.031619 ms, speedup: 7.64
transpose_smem_bcf_packed      mean time: 0.034380 ms, speedup: 7.03
####################################################################################################
n: 8192, m: 1024
torch                          mean time: 0.610348 ms
transpose_coalesced_read       mean time: 0.247641 ms, speedup: 2.46
transpose_coalesced_write      mean time: 0.249475 ms, speedup: 2.45
transpose_smem                 mean time: 0.226577 ms, speedup: 2.69
transpose_smem_bcf             mean time: 0.207867 ms, speedup: 2.94
transpose_smem_bcf_packed      mean time: 0.208608 ms, speedup: 2.93
####################################################################################################
n: 8192, m: 2048
torch                          mean time: 1.191077 ms
transpose_coalesced_read       mean time: 0.515059 ms, speedup: 2.31
transpose_coalesced_write      mean time: 0.489701 ms, speedup: 2.43
transpose_smem                 mean time: 0.435325 ms, speedup: 2.74
transpose_smem_bcf             mean time: 0.444186 ms, speedup: 2.68
transpose_smem_bcf_packed      mean time: 0.438978 ms, speedup: 2.71
####################################################################################################
n: 8192, m: 4096
torch                          mean time: 2.246580 ms
transpose_coalesced_read       mean time: 1.205594 ms, speedup: 1.86
transpose_coalesced_write      mean time: 1.002996 ms, speedup: 2.24
transpose_smem                 mean time: 0.837933 ms, speedup: 2.68
transpose_smem_bcf             mean time: 0.894996 ms, speedup: 2.51
transpose_smem_bcf_packed      mean time: 0.859038 ms, speedup: 2.62
####################################################################################################
n: 8192, m: 8192
torch                          mean time: 5.037556 ms
transpose_coalesced_read       mean time: 2.137086 ms, speedup: 2.36
transpose_coalesced_write      mean time: 2.018606 ms, speedup: 2.50
transpose_smem                 mean time: 1.877650 ms, speedup: 2.68
transpose_smem_bcf             mean time: 1.786840 ms, speedup: 2.82
transpose_smem_bcf_packed      mean time: 1.675741 ms, speedup: 3.01
```
