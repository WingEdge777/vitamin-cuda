# sgemm

## 说明

sgemm kernel

- [x] hgemm_cublas bf16/fp16 版
- [x] hgemm_naive bf16/fp16 版
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
torch                                    mean time: 4.442411 ms, 30.94 tflops
hgemm_cublas                             mean time: 4.468129 ms, speedup: 0.99, tflops: 30.76
hgemm_naive                              mean time: 5.522855 ms, speedup: 0.80, tflops: 24.89
```
