# radix sort

## Overview

radix sort kernels.

- [x] cub_sort — BF16
- [ ] sort — BF16
- [x] pytorch op bindings && diff check

## Run tests

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### Sample output

```bash

```
