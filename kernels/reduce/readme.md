# reduce_sum

## 说明

reduce_sum kernel

- [] reduce_sum fp32/fp16 版
- [] pytorch op bindings && diff check

## 测试

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### 输出

```bash

```
