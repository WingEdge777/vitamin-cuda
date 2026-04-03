# sgemm

## 说明

hgemm kernel

sm 120 kernels

- [x] hgemm_bcf_dbf_rw baseline: cp.async + ldmatrix + mma (+ double buffer + coalesced gmem)
- [x] cp.async + ldmatrix + mma + 2/3 stages buffer + coalesced gmem
- [x] tma read + ldmatrix + mma: one block, 3 stages, 128x128x64, double buffer register
- [x] tma read + ldmatrix + mma: two blocks, 3 stages, 128x128x32, double buffer register
- [] tma read/write + ldmatrix + mma
- [x] pytorch op bindings && diff check

## 测试

```bash
nvidia-smi -q -d SUPPORTED_CLOCKS
nvidia-smi -lgc 3050  # 锁定核心频率范围 (Lock Graphics Clocks)
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
nvidia-smi -rgc  # 重置核心频率
```

经过调整，tma + ldmatrix + mma 的kernel跑通了，但是实测性能也就一般。

## 输出

```yaml
####################################################################################################
n: 4096, m: 4096, k: 4096
torch                                    mean time: 3.996972 ms, 34.39 tflops
hgemm_cublas                             mean time: 4.189541 ms, speedup: 0.95, tflops: 32.81
hgemm_bcf_dbf_rw                         mean time: 4.098958 ms, speedup: 0.98, tflops: 33.53
hgemm_k_stages                           mean time: 4.171379 ms, speedup: 0.96, tflops: 32.95
hgemm_tma_r_k_stages_64                  mean time: 4.273732 ms, speedup: 0.94, tflops: 32.16
hgemm_tma_r_k_stages_32                  mean time: 4.069529 ms, speedup: 0.98, tflops: 33.77
```
