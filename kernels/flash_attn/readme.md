# flash_attn

## Overview

flash attention on SM120 (Blackwell-class) targets.

- [x] fmha: sigle buffer (BMxBN = 64x64, better for small seq_len)
- [x] pytorch op bindings && diff check

Here is a breakdown of the technical decisions under the hood:

- BMxBN = 64x64 Tiling: Optimized for sequence chunks with head_dim=128.
- Raw TMA via PTX: Splitting head_dim into two chunks (issuing two TMA copies for each tile).
- Deferred Lazy Rescaling: Reduces overhead for online softmax.

## Run tests

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

## Sample output

```yaml
####################################################################################################
prefill, batch:  1, seq: 512, head: 32, dim: 128
torch                                    mean time: 0.121095 ms, 17.73 tflops
fmha_tma_128                             mean time: 0.098975 ms, speedup: 1.22, tflops: 21.70
####################################################################################################
prefill, batch:  1, seq: 1024, head: 32, dim: 128
torch                                    mean time: 0.399812 ms, 21.48 tflops
fmha_tma_128                             mean time: 0.352873 ms, speedup: 1.13, tflops: 24.34
####################################################################################################
prefill, batch:  1, seq: 2048, head: 32, dim: 128
torch                                    mean time: 1.355930 ms, 25.34 tflops
fmha_tma_128                             mean time: 1.293091 ms, speedup: 1.05, tflops: 26.57
####################################################################################################
prefill, batch:  1, seq: 4096, head: 32, dim: 128
torch                                    mean time: 4.983902 ms, 27.58 tflops
fmha_tma_128                             mean time: 4.891181 ms, speedup: 1.02, tflops: 28.10
####################################################################################################
prefill, batch:  1, seq: 8192, head: 32, dim: 128
torch                                    mean time: 18.248972 ms, 30.13 tflops
fmha_tma_128                             mean time: 17.612578 ms, speedup: 1.04, tflops: 31.21
```
