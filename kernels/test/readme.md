# gemm

## 说明

gemm kernel

- [x] sgemm_cublas fp32/tf32 版
- [x] sgemm_tiling (向量化读写 + block tiling共享内存版)
- [x] sgemm_at_tiling (向量化读写 + a矩阵转置写入smem, 4-way 写入冲突, 内层循环float4读取)
- [x] sgemm_at_bcf_swizzling (向量化读写 + at + swizzle， 无冲突版)
- [x] sgemm_at_bcf_swizzling_rw (向量化读写 + at + swizzle + c写回事务合并)
- [x] sgemm_at_bcf_swizzling_dbf_rw(向量化读写 + at + swizzle + c写回事务合并 + double buffer流水线, 超越cuBLAS)
- [x] pytorch op bindings && diff check

## 测试

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### 输出

### 4096x4096x4096

```bash
####################################################################################################
n: 4096, m: 4096, k: 4096
torch                          mean time: 14.974799 ms
sgemm_cublas                   mean time: 14.523163 ms, speedup: 1.03
sgemm_tiling                   mean time: 18.760985 ms, speedup: 0.80
sgemm_at_tiling                mean time: 16.436968 ms, speedup: 0.91
sgemm_at_bcf_swizzling         mean time: 15.706529 ms, speedup: 0.95
sgemm_at_bcf_swizzling_rw      mean time: 15.522802 ms, speedup: 0.96
sgemm_at_bcf_swizzling_dbf_rw  mean time: 14.193397 ms, speedup: 1.06
sgemm_cublas_tf32              mean time:  8.798057 ms, speedup: 1.70
```

### all

```bash
####################################################################################################
m: 512, n: 256, k: 256
torch                          mean time: 0.062467 ms
sgemm_cublas                   mean time: 0.040665 ms, speedup: 1.54
sgemm                          mean time: 0.033703 ms, speedup: 1.85
sgemm_bcf                      mean time: 0.031690 ms, speedup: 1.97
sgemm_cublas_tf32              mean time: 0.012136 ms, speedup: 5.15
####################################################################################################
m: 512, n: 512, k: 256
torch                          mean time: 0.026130 ms
sgemm_cublas                   mean time: 0.025757 ms, speedup: 1.01
sgemm                          mean time: 0.033650 ms, speedup: 0.78
sgemm_bcf                      mean time: 0.057856 ms, speedup: 0.45
sgemm_cublas_tf32              mean time: 0.018859 ms, speedup: 1.39
####################################################################################################
m: 512, n: 256, k: 512
torch                          mean time: 0.056480 ms
sgemm_cublas                   mean time: 0.031005 ms, speedup: 1.82
sgemm                          mean time: 0.058627 ms, speedup: 0.96
sgemm_bcf                      mean time: 0.062091 ms, speedup: 0.91
sgemm_cublas_tf32              mean time: 0.020900 ms, speedup: 2.70
####################################################################################################
m: 512, n: 512, k: 512
torch                          mean time: 0.058759 ms
sgemm_cublas                   mean time: 0.038631 ms, speedup: 1.52
sgemm                          mean time: 0.086716 ms, speedup: 0.68
sgemm_bcf                      mean time: 0.055433 ms, speedup: 1.06
sgemm_cublas_tf32              mean time: 0.025164 ms, speedup: 2.34
####################################################################################################
m: 1024, n: 512, k: 512
torch                          mean time: 0.072586 ms
sgemm_cublas                   mean time: 0.070146 ms, speedup: 1.03
sgemm                          mean time: 0.105271 ms, speedup: 0.69
sgemm_bcf                      mean time: 0.116874 ms, speedup: 0.62
sgemm_cublas_tf32              mean time: 0.046662 ms, speedup: 1.56
####################################################################################################
m: 1024, n: 1024, k: 512
torch                          mean time: 0.152143 ms
sgemm_cublas                   mean time: 0.149413 ms, speedup: 1.02
sgemm                          mean time: 0.188179 ms, speedup: 0.81
sgemm_bcf                      mean time: 0.143108 ms, speedup: 1.06
sgemm_cublas_tf32              mean time: 0.075008 ms, speedup: 2.03
####################################################################################################
m: 1024, n: 512, k: 1024
torch                          mean time: 0.152300 ms
sgemm_cublas                   mean time: 0.158801 ms, speedup: 0.96
sgemm                          mean time: 0.225028 ms, speedup: 0.68
sgemm_bcf                      mean time: 0.208690 ms, speedup: 0.73
sgemm_cublas_tf32              mean time: 0.095371 ms, speedup: 1.60
####################################################################################################
m: 1024, n: 1024, k: 1024
torch                          mean time: 0.311959 ms
sgemm_cublas                   mean time: 0.280129 ms, speedup: 1.11
sgemm                          mean time: 0.343944 ms, speedup: 0.91
sgemm_bcf                      mean time: 0.327775 ms, speedup: 0.95
sgemm_cublas_tf32              mean time: 0.166107 ms, speedup: 1.88
####################################################################################################
m: 4096, n: 1024, k: 1024
torch                          mean time: 1.042517 ms
sgemm_cublas                   mean time: 1.080616 ms, speedup: 0.96
sgemm                          mean time: 1.178622 ms, speedup: 0.88
sgemm_bcf                      mean time: 1.149699 ms, speedup: 0.91
sgemm_cublas_tf32              mean time: 0.634827 ms, speedup: 1.64
####################################################################################################
m: 4096, n: 4096, k: 1024
torch                          mean time: 4.542118 ms
sgemm_cublas                   mean time: 4.757253 ms, speedup: 0.95
sgemm                          mean time: 4.549831 ms, speedup: 1.00
sgemm_bcf                      mean time: 4.394182 ms, speedup: 1.03
sgemm_cublas_tf32              mean time: 2.476518 ms, speedup: 1.83
####################################################################################################
m: 4096, n: 1024, k: 4096
torch                          mean time: 4.107055 ms
sgemm_cublas                   mean time: 4.653569 ms, speedup: 0.88
sgemm                          mean time: 4.289510 ms, speedup: 0.96
sgemm_bcf                      mean time: 4.529149 ms, speedup: 0.91
sgemm_cublas_tf32              mean time: 2.390973 ms, speedup: 1.72
####################################################################################################
m: 4096, n: 4096, k: 4096
torch                          mean time: 16.741930 ms
sgemm_cublas                   mean time: 16.232075 ms, speedup: 1.03
sgemm                          mean time: 17.583192 ms, speedup: 0.95
sgemm_bcf                      mean time: 16.882191 ms, speedup: 0.99
sgemm_cublas_tf32              mean time: 8.862989 ms, speedup: 1.89
####################################################################################################
m: 8192, n: 4096, k: 4096
torch                          mean time: 28.809838 ms
sgemm_cublas                   mean time: 29.222592 ms, speedup: 0.99
sgemm                          mean time: 34.324652 ms, speedup: 0.84
sgemm_bcf                      mean time: 32.647704 ms, speedup: 0.88
sgemm_cublas_tf32              mean time: 17.683384 ms, speedup: 1.63
####################################################################################################
m: 8192, n: 8192, k: 4096
torch                          mean time: 58.028348 ms
sgemm_cublas                   mean time: 57.443838 ms, speedup: 1.01
sgemm                          mean time: 70.755544 ms, speedup: 0.82
sgemm_bcf                      mean time: 69.478713 ms, speedup: 0.84
sgemm_cublas_tf32              mean time: 37.227454 ms, speedup: 1.56
####################################################################################################
m: 8192, n: 4096, k: 8192
torch                          mean time: 72.159470 ms
sgemm_cublas                   mean time: 73.127017 ms, speedup: 0.99
sgemm                          mean time: 76.206392 ms, speedup: 0.95
sgemm_bcf                      mean time: 73.002593 ms, speedup: 0.99
sgemm_cublas_tf32              mean time: 37.799734 ms, speedup: 1.91
####################################################################################################
m: 8192, n: 8192, k: 8192
torch                          mean time: 127.732361 ms
sgemm_cublas                   mean time: 125.815921 ms, speedup: 1.02
sgemm                          mean time: 153.963272 ms, speedup: 0.83
sgemm_bcf                      mean time: 148.995780 ms, speedup: 0.86
sgemm_cublas_tf32              mean time: 78.690780 ms, speedup: 1.62
```
