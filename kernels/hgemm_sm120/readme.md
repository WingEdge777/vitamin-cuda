# sgemm

## 说明

hgemm kernel

sm 120 kernels

- [x] hgemm_bcf_dbf_rw baseline: cp.async + ldmatrix + mma (+ double buffer + coalesced gmem)
- [x] cp.async + ldmatrix + mma + 2/3 stages buffer + coalesced gmem
- [] tma read + ldmatrix + mma
- [] tma read/write + ldmatrix + mma
- [x] pytorch op bindings && diff check

## 测试

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

## 输出

```yaml

```
