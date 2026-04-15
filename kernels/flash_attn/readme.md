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
torch                                    mean time: 0.117084 ms, 18.34 tflops
fmha_tma_128                             mean time: 0.101642 ms, speedup: 1.15, tflops: 21.13
####################################################################################################
prefill, batch:  1, seq: 1024, head: 32, dim: 128
torch                                    mean time: 0.386178 ms, 22.24 tflops
fmha_tma_128                             mean time: 0.381567 ms, speedup: 1.01, tflops: 22.51
####################################################################################################
prefill, batch:  1, seq: 2048, head: 32, dim: 128
torch                                    mean time: 1.308728 ms, 26.25 tflops
fmha_tma_128                             mean time: 1.304497 ms, speedup: 1.00, tflops: 26.34
####################################################################################################
prefill, batch:  1, seq: 4096, head: 32, dim: 128
torch                                    mean time: 4.823970 ms, 28.49 tflops
fmha_tma_128                             mean time: 4.981584 ms, speedup: 0.97, tflops: 27.59
####################################################################################################
prefill, batch:  1, seq: 8192, head: 32, dim: 128
torch                                    mean time: 18.516824 ms, 29.69 tflops
fmha_tma_128                             mean time: 18.195139 ms, speedup: 1.02, tflops: 30.21
```
