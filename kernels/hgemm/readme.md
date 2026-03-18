# sgemm

## 说明

sgemm kernel

- [x] hgemm_cublas bf16/fp16 版
- [x] hgemm_naive bf16/fp16 版 (ldmatrix + mma)
- [x] hgemm_bcf bf16/fp16 版 (ldmatrix + mma, As/Bs swizzle bcf, 95~99% cuBLAS' performance)
- [x] hgemm_bcf_dbf bf16/fp16 版 (ldmatrix + mma, As/Bs swizzle bcf, double buffer, outperforming cuBLAS)
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
torch                                    mean time: 4.139693 ms, 33.20 tflops
hgemm_cublas                             mean time: 4.265322 ms, speedup: 0.97, tflops: 32.22
hgemm_naive                              mean time: 5.176603 ms, speedup: 0.80, tflops: 26.55
hgemm_bcf                                mean time: 4.259930 ms, speedup: 0.97, tflops: 32.26
hgemm_bcf_dbf                            mean time: 4.055756 ms, speedup: 1.02, tflops: 33.89
```
