# softmax

## 说明

softmax kernel

- [x] safe online softmax fp32 版 ()
- [x] safe online softmax fp32x4 版 ()
- [x] pytorch op bindings && diff check

## 测试

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### 输出

```bash

```
