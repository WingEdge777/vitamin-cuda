# radix sort

## Overview

radix sort kernels.

- [ ] cub_sort — BF16
- [ ] radix_sort — BF16
- [x] pytorch op bindings && diff check

## Run tests

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### Sample output

```bash

```
