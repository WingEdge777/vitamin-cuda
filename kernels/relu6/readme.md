# relu6

## Overview

ReLU6 kernels.

- [x] relu6 — FP32 / FP16
- [x] relu6_fp16x2 — vectorized FP16
- [x] relu6_fp16x8 — vectorized FP16
- [x] relu6_fp16x8_packed — vectorized FP16, packed r/w
- [x] pytorch op bindings && diff check

## Run tests

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### Sample output

```bash

```
