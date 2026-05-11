# triton_add

## Overview

- [x] triton add
- [x] tilelang add

## Run tests

```bash
python test.py
```

## Sample output

```yaml
####################################################################################################
vector add, n: 1024
torch                                    mean time: 0.007588 ms
triton                                   mean time: 0.017937 ms, speedup: 0.42
tilelang                                 mean time: 0.006911 ms, speedup: 1.10
elementwise_add_fp32x4                   mean time: 0.007327 ms, speedup: 1.04
torch                                    mean time: 0.007537 ms
triton                                   mean time: 0.010621 ms, speedup: 0.71
tilelang                                 mean time: 0.007488 ms, speedup: 1.01
elementwise_add_fp16x8_packed            mean time: 0.007421 ms, speedup: 1.02
####################################################################################################
vector add, n: 4096
torch                                    mean time: 0.011768 ms
triton                                   mean time: 0.011026 ms, speedup: 1.07
tilelang                                 mean time: 0.007121 ms, speedup: 1.65
elementwise_add_fp32x4                   mean time: 0.007336 ms, speedup: 1.60
torch                                    mean time: 0.009571 ms
triton                                   mean time: 0.018703 ms, speedup: 0.51
tilelang                                 mean time: 0.009615 ms, speedup: 1.00
elementwise_add_fp16x8_packed            mean time: 0.008063 ms, speedup: 1.19
####################################################################################################
vector add, n: 32768
torch                                    mean time: 0.023158 ms
triton                                   mean time: 0.020407 ms, speedup: 1.13
tilelang                                 mean time: 0.007737 ms, speedup: 2.99
elementwise_add_fp32x4                   mean time: 0.007555 ms, speedup: 3.07
torch                                    mean time: 0.009079 ms
triton                                   mean time: 0.012769 ms, speedup: 0.71
tilelang                                 mean time: 0.007803 ms, speedup: 1.16
elementwise_add_fp16x8_packed            mean time: 0.007428 ms, speedup: 1.22
####################################################################################################
vector add, n: 1048576
torch                                    mean time: 0.026512 ms
triton                                   mean time: 0.036234 ms, speedup: 0.73
tilelang                                 mean time: 0.020966 ms, speedup: 1.26
elementwise_add_fp32x4                   mean time: 0.013102 ms, speedup: 2.02
torch                                    mean time: 0.011364 ms
triton                                   mean time: 0.015189 ms, speedup: 0.75
tilelang                                 mean time: 0.020856 ms, speedup: 0.54
elementwise_add_fp16x8_packed            mean time: 0.010305 ms, speedup: 1.10
####################################################################################################
vector add, n: 4194304
torch                                    mean time: 0.126025 ms
triton                                   mean time: 0.127259 ms, speedup: 0.99
tilelang                                 mean time: 0.148157 ms, speedup: 0.85
elementwise_add_fp32x4                   mean time: 0.130473 ms, speedup: 0.97
torch                                    mean time: 0.022681 ms
triton                                   mean time: 0.034263 ms, speedup: 0.66
tilelang                                 mean time: 0.059784 ms, speedup: 0.38
elementwise_add_fp16x8_packed            mean time: 0.023995 ms, speedup: 0.95
####################################################################################################
vector add, n: 16777216
torch                                    mean time: 0.636325 ms
triton                                   mean time: 0.637536 ms, speedup: 1.00
tilelang                                 mean time: 0.631125 ms, speedup: 1.01
elementwise_add_fp32x4                   mean time: 0.628436 ms, speedup: 1.01
torch                                    mean time: 0.315659 ms
triton                                   mean time: 0.315830 ms, speedup: 1.00
tilelang                                 mean time: 0.350582 ms, speedup: 0.90
elementwise_add_fp16x8_packed            mean time: 0.315007 ms, speedup: 1.00
```
