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
torch.compile                            mean time: 0.458745 ms, 292.61 GB/s
flash_decode_tma_128                     mean time: 0.446513 ms, speedup: 1.03, GB/s: 300.63
####################################################################################################
prefill, kv seq: 16384, head: 32, dim: 128
torch.compile                            mean time: 0.878726 ms, 305.50 GB/s
flash_decode_tma_128                     mean time: 0.778402 ms, speedup: 1.13, GB/s: 344.88
####################################################################################################
prefill, kv seq: 10240, head: 32, dim: 128
torch.compile                            mean time: 0.611377 ms, 274.44 GB/s
flash_decode_tma_128                     mean time: 0.520998 ms, speedup: 1.17, GB/s: 322.05
####################################################################################################
prefill, kv seq: 65536, head: 32, dim: 128
torch.compile                            mean time: 2.937899 ms, 365.49 GB/s
flash_decode_tma_128                     mean time: 2.920613 ms, speedup: 1.01, GB/s: 367.65
####################################################################################################
prefill, kv seq: 131072, head: 32, dim: 128
torch.compile                            mean time: 5.873622 ms, 365.62 GB/s
flash_decode_tma_128                     mean time: 5.720886 ms, speedup: 1.03, GB/s: 375.38
####################################################################################################
prefill, kv seq: 131073, head: 32, dim: 128
torch.compile                            mean time: 5.959847 ms, 360.33 GB/s
flash_decode_tma_128                     mean time: 5.718791 ms, speedup: 1.04, GB/s: 375.52
```
