# sgemm

## 说明

sgemm kernel

- [x] hgemm_cublas bf16/fp16 版
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
torch                                    mean time: 4.006988 ms, 34.30 tflops
hgemm_cublas                             mean time: 4.144308 ms, speedup: 0.97, tflops: 33.16
torch                                    mean time: 4.601224 ms, 29.87 tflops
hgemm_cublas                             mean time: 4.816182 ms, speedup: 0.96, tflops: 28.54
```
