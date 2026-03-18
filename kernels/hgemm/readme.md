# sgemm

## 说明

sgemm kernel

- [x] hgemm_cublas bf16/fp16 版
- [x] hgemm_naive bf16/fp16 版 (ldmatrix + mma)
- [x] hgemm_bcf bf16/fp16 版 (ldmatrix + mma, As/Bs swizzle bcf, 95~99% cuBLAS' performance)
- [x] pytorch op bindings && diff check

## 测试

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

## 输出

```yaml
####################################################################################################
n: 4096, m: 4096, k: 4096
torch                                    mean time: 4.137898 ms, 33.21 tflops
hgemm_cublas                             mean time: 4.232377 ms, speedup: 0.98, tflops: 32.47
hgemm_naive                              mean time: 5.141741 ms, speedup: 0.80, tflops: 26.73
hgemm_bcf                                mean time: 4.235186 ms, speedup: 0.98, tflops: 32.45
```
