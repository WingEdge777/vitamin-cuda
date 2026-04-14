# flash_attn

## Overview

flash attention on SM120 (Blackwell-class) targets.

- [x] fmha: sigle buffer (BMxBN = 64x64, better for small seq_len)
- [x] pytorch op bindings && diff check

## Run tests

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

## Sample output

```yaml
####################################################################################################
prefill, batch:  1, seq: 512, head: 32, dim: 128
torch                                    mean time: 0.117051 ms, 18.35 tflops
fmha_tma_128                             mean time: 0.101989 ms, speedup: 1.15, tflops: 21.06
####################################################################################################
prefill, batch:  1, seq: 1024, head: 32, dim: 128
torch                                    mean time: 0.410450 ms, 20.93 tflops
fmha_tma_128                             mean time: 0.413180 ms, speedup: 0.99, tflops: 20.79
####################################################################################################
prefill, batch:  1, seq: 2048, head: 32, dim: 128
torch                                    mean time: 1.376297 ms, 24.97 tflops
fmha_tma_128                             mean time: 1.377038 ms, speedup: 1.00, tflops: 24.95
####################################################################################################
prefill, batch:  1, seq: 4096, head: 32, dim: 128
torch                                    mean time: 5.215681 ms, 26.35 tflops
fmha_tma_128                             mean time: 5.198346 ms, speedup: 1.00, tflops: 26.44
```
