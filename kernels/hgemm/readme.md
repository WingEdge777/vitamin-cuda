# sgemm

## 说明

sgemm kernel

- [x] hgemm_cublas bf16/fp16 版
- [x] hgemm_naive bf16/fp16 版
- [] hgemm_bcf bf16/fp16 版
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
torch                                    mean time: 4.114296 ms, 33.41 tflops
hgemm_cublas                             mean time: 4.219122 ms, speedup: 0.98, tflops: 32.58
hgemm_naive                              mean time: 5.238569 ms, speedup: 0.79, tflops: 26.24
hgemm_bcf                                mean time: 4.333340 ms, speedup: 0.95, tflops: 31.72
```
