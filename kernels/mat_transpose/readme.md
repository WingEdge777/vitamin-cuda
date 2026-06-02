# Transpose

## Overview

Matrix transpose kernels

- [x] `transpose_coalesced_read` — coalesced global loads (input-centric)
- [x] `transpose_coalesced_write` — coalesced global stores (output-centric)
- [x] `transpose_smem` — shared-memory tiled transpose
- [x] `transpose_smem_bcf` — SMEM bank-conflict-free (padding)
- [x] `transpose_smem_packed_bcf` — SMEM bank-conflict-free, `float4` vectorized r/w
- [x] `transpose_smem_swizzled_packed` — SMEM swizzled, `float4` vectorized r/w
- [x] PyTorch op binding & correctness check

## Build & Test

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### Benchmark Results

```bash
####################################################################################################
n: 256, m: 256
torch                          mean time: 0.016384 ms
transpose_coalesced_read       mean time: 0.009632 ms, speedup: 1.70
transpose_coalesced_write      mean time: 0.010848 ms, speedup: 1.51
transpose_smem                 mean time: 0.009440 ms, speedup: 1.74
transpose_smem_bcf             mean time: 0.008032 ms, speedup: 2.04
transpose_smem_packed_bcf      mean time: 0.007984 ms, speedup: 2.05
transpose_smem_swizzled_packed mean time: 0.007744 ms, speedup: 2.12
####################################################################################################
n: 256, m: 512
torch                          mean time: 0.018240 ms
transpose_coalesced_read       mean time: 0.011520 ms, speedup: 1.58
transpose_coalesced_write      mean time: 0.011376 ms, speedup: 1.60
transpose_smem                 mean time: 0.009280 ms, speedup: 1.97
transpose_smem_bcf             mean time: 0.009056 ms, speedup: 2.01
transpose_smem_packed_bcf      mean time: 0.009184 ms, speedup: 1.99
transpose_smem_swizzled_packed mean time: 0.009152 ms, speedup: 1.99
####################################################################################################
n: 256, m: 1024
torch                          mean time: 0.023776 ms
transpose_coalesced_read       mean time: 0.017184 ms, speedup: 1.38
transpose_coalesced_write      mean time: 0.015120 ms, speedup: 1.57
transpose_smem                 mean time: 0.013248 ms, speedup: 1.79
transpose_smem_bcf             mean time: 0.011328 ms, speedup: 2.10
transpose_smem_packed_bcf      mean time: 0.010512 ms, speedup: 2.26
transpose_smem_swizzled_packed mean time: 0.010784 ms, speedup: 2.20
####################################################################################################
n: 256, m: 2048
torch                          mean time: 0.038640 ms
transpose_coalesced_read       mean time: 0.025744 ms, speedup: 1.50
transpose_coalesced_write      mean time: 0.029088 ms, speedup: 1.33
transpose_smem                 mean time: 0.019808 ms, speedup: 1.95
transpose_smem_bcf             mean time: 0.013888 ms, speedup: 2.78
transpose_smem_packed_bcf      mean time: 0.014816 ms, speedup: 2.61
transpose_smem_swizzled_packed mean time: 0.014400 ms, speedup: 2.68
####################################################################################################
n: 256, m: 4096
torch                          mean time: 0.066672 ms
transpose_coalesced_read       mean time: 0.041136 ms, speedup: 1.62
transpose_coalesced_write      mean time: 0.045888 ms, speedup: 1.45
transpose_smem                 mean time: 0.031712 ms, speedup: 2.10
transpose_smem_bcf             mean time: 0.026592 ms, speedup: 2.51
transpose_smem_packed_bcf      mean time: 0.026576 ms, speedup: 2.51
transpose_smem_swizzled_packed mean time: 0.027216 ms, speedup: 2.45
####################################################################################################
n: 256, m: 8192
torch                          mean time: 0.100176 ms
transpose_coalesced_read       mean time: 0.072656 ms, speedup: 1.38
transpose_coalesced_write      mean time: 0.080336 ms, speedup: 1.25
transpose_smem                 mean time: 0.055840 ms, speedup: 1.79
transpose_smem_bcf             mean time: 0.053824 ms, speedup: 1.86
transpose_smem_packed_bcf      mean time: 0.054288 ms, speedup: 1.85
transpose_smem_swizzled_packed mean time: 0.054016 ms, speedup: 1.85
####################################################################################################
n: 512, m: 256
torch                          mean time: 0.018144 ms
transpose_coalesced_read       mean time: 0.012400 ms, speedup: 1.46
transpose_coalesced_write      mean time: 0.010976 ms, speedup: 1.65
transpose_smem                 mean time: 0.009376 ms, speedup: 1.94
transpose_smem_bcf             mean time: 0.009504 ms, speedup: 1.91
transpose_smem_packed_bcf      mean time: 0.009648 ms, speedup: 1.88
transpose_smem_swizzled_packed mean time: 0.009184 ms, speedup: 1.98
####################################################################################################
n: 512, m: 512
torch                          mean time: 0.022416 ms
transpose_coalesced_read       mean time: 0.014912 ms, speedup: 1.50
transpose_coalesced_write      mean time: 0.015472 ms, speedup: 1.45
transpose_smem                 mean time: 0.013472 ms, speedup: 1.66
transpose_smem_bcf             mean time: 0.010608 ms, speedup: 2.11
transpose_smem_packed_bcf      mean time: 0.010656 ms, speedup: 2.10
transpose_smem_swizzled_packed mean time: 0.010816 ms, speedup: 2.07
####################################################################################################
n: 512, m: 1024
torch                          mean time: 0.036096 ms
transpose_coalesced_read       mean time: 0.025632 ms, speedup: 1.41
transpose_coalesced_write      mean time: 0.025968 ms, speedup: 1.39
transpose_smem                 mean time: 0.019360 ms, speedup: 1.86
transpose_smem_bcf             mean time: 0.014208 ms, speedup: 2.54
transpose_smem_packed_bcf      mean time: 0.013632 ms, speedup: 2.65
transpose_smem_swizzled_packed mean time: 0.013712 ms, speedup: 2.63
####################################################################################################
n: 512, m: 2048
torch                          mean time: 0.056960 ms
transpose_coalesced_read       mean time: 0.041344 ms, speedup: 1.38
transpose_coalesced_write      mean time: 0.042720 ms, speedup: 1.33
transpose_smem                 mean time: 0.031376 ms, speedup: 1.82
transpose_smem_bcf             mean time: 0.026528 ms, speedup: 2.15
transpose_smem_packed_bcf      mean time: 0.026112 ms, speedup: 2.18
transpose_smem_swizzled_packed mean time: 0.027088 ms, speedup: 2.10
####################################################################################################
n: 512, m: 4096
torch                          mean time: 0.109136 ms
transpose_coalesced_read       mean time: 0.068000 ms, speedup: 1.60
transpose_coalesced_write      mean time: 0.082368 ms, speedup: 1.32
transpose_smem                 mean time: 0.057920 ms, speedup: 1.88
transpose_smem_bcf             mean time: 0.053824 ms, speedup: 2.03
transpose_smem_packed_bcf      mean time: 0.053920 ms, speedup: 2.02
transpose_smem_swizzled_packed mean time: 0.054416 ms, speedup: 2.01
####################################################################################################
n: 512, m: 8192
torch                          mean time: 0.183536 ms
transpose_coalesced_read       mean time: 0.139248 ms, speedup: 1.32
transpose_coalesced_write      mean time: 0.139408 ms, speedup: 1.32
transpose_smem                 mean time: 0.107328 ms, speedup: 1.71
transpose_smem_bcf             mean time: 0.105104 ms, speedup: 1.75
transpose_smem_packed_bcf      mean time: 0.104976 ms, speedup: 1.75
transpose_smem_swizzled_packed mean time: 0.104960 ms, speedup: 1.75
####################################################################################################
n: 1024, m: 256
torch                          mean time: 0.022288 ms
transpose_coalesced_read       mean time: 0.015120 ms, speedup: 1.47
transpose_coalesced_write      mean time: 0.015136 ms, speedup: 1.47
transpose_smem                 mean time: 0.013296 ms, speedup: 1.68
transpose_smem_bcf             mean time: 0.010528 ms, speedup: 2.12
transpose_smem_packed_bcf      mean time: 0.010176 ms, speedup: 2.19
transpose_smem_swizzled_packed mean time: 0.010928 ms, speedup: 2.04
####################################################################################################
n: 1024, m: 512
torch                          mean time: 0.035776 ms
transpose_coalesced_read       mean time: 0.026896 ms, speedup: 1.33
transpose_coalesced_write      mean time: 0.027616 ms, speedup: 1.30
transpose_smem                 mean time: 0.019904 ms, speedup: 1.80
transpose_smem_bcf             mean time: 0.013264 ms, speedup: 2.70
transpose_smem_packed_bcf      mean time: 0.013360 ms, speedup: 2.68
transpose_smem_swizzled_packed mean time: 0.013376 ms, speedup: 2.67
####################################################################################################
n: 1024, m: 1024
torch                          mean time: 0.058192 ms
transpose_coalesced_read       mean time: 0.042400 ms, speedup: 1.37
transpose_coalesced_write      mean time: 0.043664 ms, speedup: 1.33
transpose_smem                 mean time: 0.032256 ms, speedup: 1.80
transpose_smem_bcf             mean time: 0.026064 ms, speedup: 2.23
transpose_smem_packed_bcf      mean time: 0.025696 ms, speedup: 2.26
transpose_smem_swizzled_packed mean time: 0.025904 ms, speedup: 2.25
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.103824 ms
transpose_coalesced_read       mean time: 0.069696 ms, speedup: 1.49
transpose_coalesced_write      mean time: 0.077184 ms, speedup: 1.35
transpose_smem                 mean time: 0.058592 ms, speedup: 1.77
transpose_smem_bcf             mean time: 0.054048 ms, speedup: 1.92
transpose_smem_packed_bcf      mean time: 0.053648 ms, speedup: 1.94
transpose_smem_swizzled_packed mean time: 0.054272 ms, speedup: 1.91
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.197728 ms
transpose_coalesced_read       mean time: 0.129120 ms, speedup: 1.53
transpose_coalesced_write      mean time: 0.144320 ms, speedup: 1.37
transpose_smem                 mean time: 0.107728 ms, speedup: 1.84
transpose_smem_bcf             mean time: 0.103760 ms, speedup: 1.91
transpose_smem_packed_bcf      mean time: 0.103648 ms, speedup: 1.91
transpose_smem_swizzled_packed mean time: 0.103792 ms, speedup: 1.91
####################################################################################################
n: 1024, m: 8192
torch                          mean time: 0.501904 ms
transpose_coalesced_read       mean time: 0.302800 ms, speedup: 1.66
transpose_coalesced_write      mean time: 0.315072 ms, speedup: 1.59
transpose_smem                 mean time: 0.214624 ms, speedup: 2.34
transpose_smem_bcf             mean time: 0.211248 ms, speedup: 2.38
transpose_smem_packed_bcf      mean time: 0.210768 ms, speedup: 2.38
transpose_smem_swizzled_packed mean time: 0.212144 ms, speedup: 2.37
####################################################################################################
n: 2048, m: 256
torch                          mean time: 0.036624 ms
transpose_coalesced_read       mean time: 0.026960 ms, speedup: 1.36
transpose_coalesced_write      mean time: 0.025856 ms, speedup: 1.42
transpose_smem                 mean time: 0.021328 ms, speedup: 1.72
transpose_smem_bcf             mean time: 0.013600 ms, speedup: 2.69
transpose_smem_packed_bcf      mean time: 0.013728 ms, speedup: 2.67
transpose_smem_swizzled_packed mean time: 0.013456 ms, speedup: 2.72
####################################################################################################
n: 2048, m: 512
torch                          mean time: 0.069328 ms
transpose_coalesced_read       mean time: 0.043520 ms, speedup: 1.59
transpose_coalesced_write      mean time: 0.044656 ms, speedup: 1.55
transpose_smem                 mean time: 0.033920 ms, speedup: 2.04
transpose_smem_bcf             mean time: 0.025232 ms, speedup: 2.75
transpose_smem_packed_bcf      mean time: 0.025168 ms, speedup: 2.75
transpose_smem_swizzled_packed mean time: 0.025952 ms, speedup: 2.67
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.116544 ms
transpose_coalesced_read       mean time: 0.079824 ms, speedup: 1.46
transpose_coalesced_write      mean time: 0.083744 ms, speedup: 1.39
transpose_smem                 mean time: 0.065344 ms, speedup: 1.78
transpose_smem_bcf             mean time: 0.055424 ms, speedup: 2.10
transpose_smem_packed_bcf      mean time: 0.055424 ms, speedup: 2.10
transpose_smem_swizzled_packed mean time: 0.054944 ms, speedup: 2.12
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.225440 ms
transpose_coalesced_read       mean time: 0.138144 ms, speedup: 1.63
transpose_coalesced_write      mean time: 0.146944 ms, speedup: 1.53
transpose_smem                 mean time: 0.112160 ms, speedup: 2.01
transpose_smem_bcf             mean time: 0.102928 ms, speedup: 2.19
transpose_smem_packed_bcf      mean time: 0.104816 ms, speedup: 2.15
transpose_smem_swizzled_packed mean time: 0.103872 ms, speedup: 2.17
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.572816 ms
transpose_coalesced_read       mean time: 0.288032 ms, speedup: 1.99
transpose_coalesced_write      mean time: 0.280480 ms, speedup: 2.04
transpose_smem                 mean time: 0.227792 ms, speedup: 2.51
transpose_smem_bcf             mean time: 0.208352 ms, speedup: 2.75
transpose_smem_packed_bcf      mean time: 0.208192 ms, speedup: 2.75
transpose_smem_swizzled_packed mean time: 0.207968 ms, speedup: 2.75
####################################################################################################
n: 2048, m: 8192
torch                          mean time: 1.179840 ms
transpose_coalesced_read       mean time: 0.527712 ms, speedup: 2.24
transpose_coalesced_write      mean time: 0.494624 ms, speedup: 2.39
transpose_smem                 mean time: 0.520048 ms, speedup: 2.27
transpose_smem_bcf             mean time: 0.421296 ms, speedup: 2.80
transpose_smem_packed_bcf      mean time: 0.446128 ms, speedup: 2.64
transpose_smem_swizzled_packed mean time: 0.447296 ms, speedup: 2.64
####################################################################################################
n: 4096, m: 256
torch                          mean time: 0.067584 ms
transpose_coalesced_read       mean time: 0.044560 ms, speedup: 1.52
transpose_coalesced_write      mean time: 0.043360 ms, speedup: 1.56
transpose_smem                 mean time: 0.034352 ms, speedup: 1.97
transpose_smem_bcf             mean time: 0.026128 ms, speedup: 2.59
transpose_smem_packed_bcf      mean time: 0.027328 ms, speedup: 2.47
transpose_smem_swizzled_packed mean time: 0.027408 ms, speedup: 2.47
####################################################################################################
n: 4096, m: 512
torch                          mean time: 0.127040 ms
transpose_coalesced_read       mean time: 0.080928 ms, speedup: 1.57
transpose_coalesced_write      mean time: 0.083552 ms, speedup: 1.52
transpose_smem                 mean time: 0.062848 ms, speedup: 2.02
transpose_smem_bcf             mean time: 0.054080 ms, speedup: 2.35
transpose_smem_packed_bcf      mean time: 0.054304 ms, speedup: 2.34
transpose_smem_swizzled_packed mean time: 0.054208 ms, speedup: 2.34
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.233936 ms
transpose_coalesced_read       mean time: 0.142672 ms, speedup: 1.64
transpose_coalesced_write      mean time: 0.146480 ms, speedup: 1.60
transpose_smem                 mean time: 0.116496 ms, speedup: 2.01
transpose_smem_bcf             mean time: 0.106848 ms, speedup: 2.19
transpose_smem_packed_bcf      mean time: 0.107232 ms, speedup: 2.18
transpose_smem_swizzled_packed mean time: 0.106336 ms, speedup: 2.20
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.573520 ms
transpose_coalesced_read       mean time: 0.273200 ms, speedup: 2.10
transpose_coalesced_write      mean time: 0.265824 ms, speedup: 2.16
transpose_smem                 mean time: 0.248576 ms, speedup: 2.31
transpose_smem_bcf             mean time: 0.212000 ms, speedup: 2.71
transpose_smem_packed_bcf      mean time: 0.212608 ms, speedup: 2.70
transpose_smem_swizzled_packed mean time: 0.211600 ms, speedup: 2.71
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 1.175968 ms
transpose_coalesced_read       mean time: 0.493824 ms, speedup: 2.38
transpose_coalesced_write      mean time: 0.505440 ms, speedup: 2.33
transpose_smem                 mean time: 0.427264 ms, speedup: 2.75
transpose_smem_bcf             mean time: 0.414032 ms, speedup: 2.84
transpose_smem_packed_bcf      mean time: 0.414336 ms, speedup: 2.84
transpose_smem_swizzled_packed mean time: 0.414368 ms, speedup: 2.84
####################################################################################################
n: 4096, m: 8192
torch                          mean time: 2.233824 ms
transpose_coalesced_read       mean time: 0.948912 ms, speedup: 2.35
transpose_coalesced_write      mean time: 0.929680 ms, speedup: 2.40
transpose_smem                 mean time: 0.853856 ms, speedup: 2.62
transpose_smem_bcf             mean time: 0.866080 ms, speedup: 2.58
transpose_smem_packed_bcf      mean time: 0.909168 ms, speedup: 2.46
transpose_smem_swizzled_packed mean time: 0.908720 ms, speedup: 2.46
####################################################################################################
n: 8192, m: 256
torch                          mean time: 0.112944 ms
transpose_coalesced_read       mean time: 0.075216 ms, speedup: 1.50
transpose_coalesced_write      mean time: 0.073168 ms, speedup: 1.54
transpose_smem                 mean time: 0.059600 ms, speedup: 1.90
transpose_smem_bcf             mean time: 0.056208 ms, speedup: 2.01
transpose_smem_packed_bcf      mean time: 0.056784 ms, speedup: 1.99
transpose_smem_swizzled_packed mean time: 0.056336 ms, speedup: 2.00
####################################################################################################
n: 8192, m: 512
torch                          mean time: 0.232320 ms
transpose_coalesced_read       mean time: 0.144320 ms, speedup: 1.61
transpose_coalesced_write      mean time: 0.153088 ms, speedup: 1.52
transpose_smem                 mean time: 0.128656 ms, speedup: 1.81
transpose_smem_bcf             mean time: 0.118592 ms, speedup: 1.96
transpose_smem_packed_bcf      mean time: 0.119984 ms, speedup: 1.94
transpose_smem_swizzled_packed mean time: 0.119680 ms, speedup: 1.94
####################################################################################################
n: 8192, m: 1024
torch                          mean time: 0.565488 ms
transpose_coalesced_read       mean time: 0.266144 ms, speedup: 2.12
transpose_coalesced_write      mean time: 0.273360 ms, speedup: 2.07
transpose_smem                 mean time: 0.243152 ms, speedup: 2.33
transpose_smem_bcf             mean time: 0.223664 ms, speedup: 2.53
transpose_smem_packed_bcf      mean time: 0.223968 ms, speedup: 2.52
transpose_smem_swizzled_packed mean time: 0.224400 ms, speedup: 2.52
####################################################################################################
n: 8192, m: 2048
torch                          mean time: 1.120304 ms
transpose_coalesced_read       mean time: 0.488576 ms, speedup: 2.29
transpose_coalesced_write      mean time: 0.495488 ms, speedup: 2.26
transpose_smem                 mean time: 0.420128 ms, speedup: 2.67
transpose_smem_bcf             mean time: 0.413152 ms, speedup: 2.71
transpose_smem_packed_bcf      mean time: 0.412800 ms, speedup: 2.71
transpose_smem_swizzled_packed mean time: 0.413456 ms, speedup: 2.71
####################################################################################################
n: 8192, m: 4096
torch                          mean time: 2.207104 ms
transpose_coalesced_read       mean time: 0.925536 ms, speedup: 2.38
transpose_coalesced_write      mean time: 0.985120 ms, speedup: 2.24
transpose_smem                 mean time: 0.826784 ms, speedup: 2.67
transpose_smem_bcf             mean time: 0.833264 ms, speedup: 2.65
transpose_smem_packed_bcf      mean time: 0.832912 ms, speedup: 2.65
transpose_smem_swizzled_packed mean time: 0.829456 ms, speedup: 2.66
####################################################################################################
n: 8192, m: 8192
torch                          mean time: 4.380160 ms
transpose_coalesced_read       mean time: 1.923440 ms, speedup: 2.28
transpose_coalesced_write      mean time: 2.037584 ms, speedup: 2.15
transpose_smem                 mean time: 1.793504 ms, speedup: 2.44
transpose_smem_bcf             mean time: 1.816736 ms, speedup: 2.41
transpose_smem_packed_bcf      mean time: 1.815856 ms, speedup: 2.41
transpose_smem_swizzled_packed mean time: 1.730048 ms, speedup: 2.53
```
