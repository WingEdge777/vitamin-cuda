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
torch.compile                            mean time: 0.440589 ms, 304.67 GB/s
flash-infer                              mean time: 0.432449 ms, speedup: 1.02, GB/s: 310.40
flash_decode_tma_128                     mean time: 0.392698 ms, speedup: 1.12, GB/s: 341.83
flash_decode_tma_dbf_k_128               mean time: 0.410151 ms, speedup: 1.07, GB/s: 327.28
####################################################################################################
prefill, kv seq: 10240, head: 32, dim: 128
torch.compile                            mean time: 0.538761 ms, 311.43 GB/s
flash-infer                              mean time: 0.458090 ms, speedup: 1.18, GB/s: 366.28
flash_decode_tma_128                     mean time: 0.489902 ms, speedup: 1.10, GB/s: 342.49
flash_decode_tma_dbf_k_128               mean time: 0.475617 ms, speedup: 1.13, GB/s: 352.78
####################################################################################################
prefill, kv seq: 32768, head: 32, dim: 128
torch.compile                            mean time: 1.475492 ms, 363.87 GB/s
flash-infer                              mean time: 1.432426 ms, speedup: 1.03, GB/s: 374.81
flash_decode_tma_128                     mean time: 1.421604 ms, speedup: 1.04, GB/s: 377.66
flash_decode_tma_dbf_k_128               mean time: 1.425331 ms, speedup: 1.04, GB/s: 376.68
####################################################################################################
prefill, kv seq: 65536, head: 32, dim: 128
torch.compile                            mean time: 2.974706 ms, 360.96 GB/s
flash-infer                              mean time: 2.833499 ms, speedup: 1.05, GB/s: 378.95
flash_decode_tma_128                     mean time: 2.858499 ms, speedup: 1.04, GB/s: 375.64
flash_decode_tma_dbf_k_128               mean time: 2.868142 ms, speedup: 1.04, GB/s: 374.37
####################################################################################################
prefill, kv seq: 131072, head: 32, dim: 128
torch.compile                            mean time: 5.828940 ms, 368.42 GB/s
flash-infer                              mean time: 5.669644 ms, speedup: 1.03, GB/s: 378.77
flash_decode_tma_128                     mean time: 5.595696 ms, speedup: 1.04, GB/s: 383.78
flash_decode_tma_dbf_k_128               mean time: 5.577071 ms, speedup: 1.05, GB/s: 385.06
####################################################################################################
prefill, kv seq: 131073, head: 32, dim: 128
torch.compile                            mean time: 6.263341 ms, 342.87 GB/s
flash-infer                              mean time: 5.702792 ms, speedup: 1.10, GB/s: 376.57
flash_decode_tma_128                     mean time: 5.585575 ms, speedup: 1.12, GB/s: 384.48
flash_decode_tma_dbf_k_128               mean time: 5.581956 ms, speedup: 1.12, GB/s: 384.72
```
