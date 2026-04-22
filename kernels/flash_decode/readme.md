# flash decoding

## Overview

flash decoing on SM120 (Blackwell-class) targets.

- [x] flash_decode_tma_128
- [x] pytorch.compile op bindings && diff check

## Run tests

```bash
export torch.compile_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

## Sample output

```yaml
####################################################################################################
prefill, kv seq: 8192, head: 32, dim: 128
torch.compile                            mean time: 0.458729 ms, 292.62 GB/s
flash_decode_tma_128                     mean time: 0.409808 ms, speedup: 1.12, GB/s: 327.55
####################################################################################################
prefill, kv seq: 16384, head: 32, dim: 128
torch.compile                            mean time: 0.858288 ms, 312.78 GB/s
flash_decode_tma_128                     mean time: 0.736965 ms, speedup: 1.16, GB/s: 364.27
####################################################################################################
prefill, kv seq: 10240, head: 32, dim: 128
torch.compile                            mean time: 0.556543 ms, 301.48 GB/s
flash_decode_tma_128                     mean time: 0.522694 ms, speedup: 1.06, GB/s: 321.01
####################################################################################################
prefill, kv seq: 65536, head: 32, dim: 128
torch.compile                            mean time: 2.962036 ms, 362.51 GB/s
flash_decode_tma_128                     mean time: 2.896064 ms, speedup: 1.02, GB/s: 370.76
####################################################################################################
prefill, kv seq: 131072, head: 32, dim: 128
torch.compile                            mean time: 5.868828 ms, 365.92 GB/s
flash_decode_tma_128                     mean time: 5.731302 ms, speedup: 1.02, GB/s: 374.70
```
