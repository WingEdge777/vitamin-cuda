# sgemm

## 说明

hgemm kernel

sm 120 kernels

- [x] hgemm_bcf_dbf_rw baseline: cp.async + ldmatrix + mma (+ double buffer + coalesced gmem)
- [x] cp.async + ldmatrix + mma + 2/3 stages buffer + coalesced gmem
- [x] tma read + ldmatrix + mma
- [] tma read/write + ldmatrix + mma
- [x] pytorch op bindings && diff check

## 测试

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

经过调整，tma + ldmatrix + mma 的kernel跑通了，但是性能实在不行

## 输出

```yaml
####################################################################################################
n: 4096, m: 4096, k: 4096
torch                                    mean time: 4.340397 ms, 31.67 tflops
hgemm_cublas                             mean time: 4.422849 ms, speedup: 0.98, tflops: 31.07
hgemm_bcf_dbf_rw                         mean time: 4.312505 ms, speedup: 1.01, tflops: 31.87
hgemm_k_stages                           mean time: 4.210268 ms, speedup: 1.03, tflops: 32.64
hgemm_tma_r_k_stages                     mean time: 4.404516 ms, speedup: 0.99, tflops: 31.20
```
