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
####################################################################################################
prefill, kv seq: 8192, head: 32, dim: 128
torch                                    mean time: 0.457898 ms, 293.15 GB/s
flash_decode_tma_128                     mean time: 0.405664 ms, speedup: 1.13, GB/s: 330.90
####################################################################################################
prefill, kv seq: 16384, head: 32, dim: 128
torch                                    mean time: 0.860155 ms, 312.10 GB/s
flash_decode_tma_128                     mean time: 0.740989 ms, speedup: 1.16, GB/s: 362.29
####################################################################################################
prefill, kv seq: 10240, head: 32, dim: 128
torch                                    mean time: 0.557957 ms, 300.72 GB/s
flash_decode_tma_128                     mean time: 0.477874 ms, speedup: 1.17, GB/s: 351.11
####################################################################################################
prefill, kv seq: 65536, head: 32, dim: 128
torch                                    mean time: 2.920055 ms, 367.72 GB/s
flash_decode_tma_128                     mean time: 2.983609 ms, speedup: 0.98, GB/s: 359.89
####################################################################################################
prefill, kv seq: 131072, head: 32, dim: 128
torch                                    mean time: 5.877745 ms, 365.36 GB/s
flash_decode_tma_128                     mean time: 5.768781 ms, speedup: 1.02, GB/s: 372.26
```
