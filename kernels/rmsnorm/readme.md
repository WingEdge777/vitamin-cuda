# rmsnorm

## 说明

rmsnorm kernel

- [x] naive torch rmsnorm
- [x] rmsnorm fp32/fp16 版
- [x] rmsnorm fp32x4 版 (fp32向量化)
- [x] rmsnorm fp16x8 版 (fp16向量化, packed r/w)
- [x] pytorch op bindings && diff check

## 测试

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### 输出

```bash

```
