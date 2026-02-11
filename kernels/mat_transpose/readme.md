# transpose

## 说明

transpose kernel

- [] transpose_coalesced_read
- [] transpose_coalesced_write
- [] transpose_smem
- [] transpose_smem_bcf
- [] transpose_smem_bcf_packed
- [] pytorch op bindings && diff check

## 测试

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### 输出

```bash

```
