# flash decoding

## Overview

flash decoing on SM120 (Blackwell-class) targets.

- [x] flash_decode
- [x] pytorch op bindings && diff check

## Run tests

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

## Sample output

```yaml

```
