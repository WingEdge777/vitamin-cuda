# hardswish

## 说明

hardswish kernel

- [x] hardswish fp32/fp16 版
- [x] hardswish_fp16x2(fp16向量化)
- [x] hardswish_fp16x8(fp16向量化)
- [x] hardswish_fp16x8(fp16向量化, packed r/w)
- [x] pytorch op bindings && diff check

## 测试

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### 输出

```bash

```
