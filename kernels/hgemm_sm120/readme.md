# hgemm_sm120

## 说明

hgemm kernel

sm 120 kernels

- [x] hgemm_cublas baseline
- [x] hgemm_bcf_dbf_rw baseline: cp.async + ldmatrix + mma + double buffer + coalesced gmem r/w
- [x] hgemm_k_stages: cp.async + ldmatrix + mma + 2/3 stages buffer + coalesced gmem
- [x] hgemm_tma_r_k_stages_64: tma read + ldmatrix + mma: one block, 3 stages, 128x128x64, double buffer register
- [x] hgemm_tma_r_k_stages_32: tma read + ldmatrix + mma: two blocks, 3 stages, 128x128x32, double buffer register
- [] tma read/write + ldmatrix + mma
- [x] pytorch op bindings && diff check

## 测试

```bash
nvidia-smi -q -d SUPPORTED_CLOCKS
nvidia-smi -lgc 3050  # 锁定核心频率范围 (Lock Graphics Clocks)
nvidia-smi -lmc 12001  # 锁定显存频率 (Lock Memory Clocks)
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
nvidia-smi -rgc  # 重置核心频率
nvidia-smi -rmc  # 重置显存频率
```

经过调整，tma + ldmatrix + mma 的kernel跑通了，但是实测性能也就一般。

## 输出

```yaml
####################################################################################################
n: 4096, m: 4096, k: 4096
torch                                    mean time: 4.011336 ms, 34.26 tflops
hgemm_cublas                             mean time: 4.258727 ms, speedup: 0.94, tflops: 32.27
hgemm_bcf_dbf_rw                         mean time: 4.042202 ms, speedup: 0.99, tflops: 34.00
hgemm_k_stages                           mean time: 4.131343 ms, speedup: 0.97, tflops: 33.27
hgemm_tma_r_k_stages_64                  mean time: 4.287896 ms, speedup: 0.94, tflops: 32.05
hgemm_tma_r_k_stages_32                  mean time: 4.005909 ms, speedup: 1.00, tflops: 34.31
```
