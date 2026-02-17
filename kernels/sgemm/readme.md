# gemm

## 说明



gemm kernel

- [x] sgemm_cublas fp32/tf32 版
- [x] pytorch op bindings && diff check

## 测试

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### 输出

```bash
```
