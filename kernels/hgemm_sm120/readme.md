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
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

经过调整，tma + ldmatrix + mma 的kernel跑通了，但是实测性能也就一般。

## 输出

```yaml
####################################################################################################
n: 4096, m: 4096, k: 4096
torch                                    mean time: 4.182673 ms, 32.86 tflops
hgemm_cublas                             mean time: 4.252483 ms, speedup: 0.98, tflops: 32.32
hgemm_bcf_dbf_rw                         mean time: 4.203456 ms, speedup: 1.00, tflops: 32.70
hgemm_k_stages                           mean time: 4.203158 ms, speedup: 1.00, tflops: 32.70
hgemm_tma_r_k_stages_64                  mean time: 4.331222 ms, speedup: 0.97, tflops: 31.73
hgemm_tma_r_k_stages_32                  mean time: 4.129223 ms, speedup: 1.01, tflops: 33.28
```
