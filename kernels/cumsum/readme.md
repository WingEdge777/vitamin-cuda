# cmsum

## Overview

cmsum kernels.

- [] naive Torch cmsum
- [] cmsum — FP32
- [] pytorch op bindings && diff check

## Run tests

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### Sample output

```bash

```
