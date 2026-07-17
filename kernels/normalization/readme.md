# norm 1d

## Overview

norm kernels calculate y = (x - mean) / std

- [x] naive Torch norm
- [] norm — FP32
- [] norm — FP32x4
- [x] pytorch op bindings && diff check

## Run tests

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### Sample output

```bash

```
