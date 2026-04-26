# flash decoding

## Overview

flash decoing on SM120 (Blackwell-class) targets.

- [x] flash_decode_tma_128
- [x] flash_decode_tma_dbf_k
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
torch.compile                            mean time: 0.454655 ms, 295.24 GB/s
flash-infer                              mean time: 0.403291 ms, speedup: 1.13, GB/s: 332.85
flash_decode_tma_128                     mean time: 0.408378 ms, speedup: 1.11, GB/s: 328.70
flash_decode_tma_dbf_k_128               mean time: 0.366698 ms, speedup: 1.24, GB/s: 366.06
####################################################################################################
prefill, kv seq: 16384, head: 32, dim: 128
torch.compile                            mean time: 0.872882 ms, 307.55 GB/s
flash-infer                              mean time: 0.784423 ms, speedup: 1.11, GB/s: 342.23
flash_decode_tma_128                     mean time: 0.735274 ms, speedup: 1.19, GB/s: 365.10
flash_decode_tma_dbf_k_128               mean time: 0.733273 ms, speedup: 1.19, GB/s: 366.10
####################################################################################################
prefill, kv seq: 32768, head: 32, dim: 128
torch.compile                            mean time: 1.507921 ms, 356.04 GB/s
flash-infer                              mean time: 1.499479 ms, speedup: 1.01, GB/s: 358.05
flash_decode_tma_128                     mean time: 1.495797 ms, speedup: 1.01, GB/s: 358.93
flash_decode_tma_dbf_k_128               mean time: 1.455790 ms, speedup: 1.04, GB/s: 368.79
####################################################################################################
prefill, kv seq: 65536, head: 32, dim: 128
torch.compile                            mean time: 2.980080 ms, 360.31 GB/s
flash-infer                              mean time: 2.897006 ms, speedup: 1.03, GB/s: 370.64
flash_decode_tma_128                     mean time: 2.856871 ms, speedup: 1.04, GB/s: 375.85
flash_decode_tma_dbf_k_128               mean time: 2.849400 ms, speedup: 1.05, GB/s: 376.84
####################################################################################################
prefill, kv seq: 131072, head: 32, dim: 128
torch.compile                            mean time: 6.044398 ms, 355.29 GB/s
flash-infer                              mean time: 5.751600 ms, speedup: 1.05, GB/s: 373.37
flash_decode_tma_128                     mean time: 5.663495 ms, speedup: 1.07, GB/s: 379.18
flash_decode_tma_dbf_k_128               mean time: 5.736955 ms, speedup: 1.05, GB/s: 374.33
####################################################################################################
prefill, kv seq: 131073, head: 32, dim: 128
torch.compile                            mean time: 6.466227 ms, 332.11 GB/s
flash-infer                              mean time: 6.117131 ms, speedup: 1.06, GB/s: 351.07
flash_decode_tma_128                     mean time: 5.701174 ms, speedup: 1.13, GB/s: 376.68
flash_decode_tma_dbf_k_128               mean time: 5.695415 ms, speedup: 1.14, GB/s: 377.06
```
