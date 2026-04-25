# flash decoding

## Overview

flash decoing on SM120 (Blackwell-class) targets.

- [x] flash_decode_tma_128
- [ ] flash_decode_tma_dbf_k
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
torch.compile                            mean time: 0.451540 ms, 297.28 GB/s
flash_decode_tma_128                     mean time: 0.415926 ms, speedup: 1.09, GB/s: 322.74
####################################################################################################
prefill, kv seq: 10240, head: 32, dim: 128
torch.compile                            mean time: 0.555602 ms, 301.99 GB/s
flash_decode_tma_128                     mean time: 0.497295 ms, speedup: 1.12, GB/s: 337.40
####################################################################################################
prefill, kv seq: 32768, head: 32, dim: 128
torch.compile                            mean time: 1.522456 ms, 352.65 GB/s
flash_decode_tma_128                     mean time: 1.466595 ms, speedup: 1.04, GB/s: 366.08
####################################################################################################
prefill, kv seq: 65536, head: 32, dim: 128
torch.compile                            mean time: 2.971774 ms, 361.32 GB/s
flash_decode_tma_128                     mean time: 2.884984 ms, speedup: 1.03, GB/s: 372.19
####################################################################################################
prefill, kv seq: 131072, head: 32, dim: 128
torch.compile                            mean time: 5.808621 ms, 369.71 GB/s
flash_decode_tma_128                     mean time: 5.654609 ms, speedup: 1.03, GB/s: 379.78
####################################################################################################
prefill, kv seq: 131073, head: 32, dim: 128
torch.compile                            mean time: 5.896532 ms, 364.20 GB/s
flash_decode_tma_128                     mean time: 5.656408 ms, speedup: 1.04, GB/s: 379.66
```
