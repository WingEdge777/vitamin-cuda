# dsl_add

## Overview

- [x] triton add
- [x] tilelang add

## Run tests

```bash
python test.py
```

## Sample output

## special cases test

```yaml
####################################################################################################
vector add, n: 16777216
torch                                    mean time: 0.308125 ms
triton                                   mean time: 0.295915 ms, speedup: 1.04
tilelang                                 mean time: 0.343332 ms, speedup: 0.90
tilelang_vectorized                      mean time: 0.295918 ms, speedup: 1.04
tilelang_vectorized_eager                mean time: 0.296376 ms, speedup: 1.04
elementwise_add_fp16x8_packed            mean time: 0.295292 ms, speedup: 1.04
####################################################################################################
vector add, n: 16777217
torch                                    mean time: 0.295970 ms
triton                                   mean time: 0.295789 ms, speedup: 1.00
tilelang                                 mean time: 0.337083 ms, speedup: 0.88
tilelang_vectorized                      mean time: 0.295900 ms, speedup: 1.00
tilelang_vectorized_eager                mean time: 0.297782 ms, speedup: 0.99
elementwise_add_fp16x8_packed            mean time: 0.296227 ms, speedup: 1.00
```

### full benchmark

```yaml
####################################################################################################
vector add, n: 1024
torch                                    mean time: 0.014939 ms
triton                                   mean time: 0.009584 ms, speedup: 1.56
tilelang                                 mean time: 0.015537 ms, speedup: 0.96
elementwise_add_fp32x4                   mean time: 0.008070 ms, speedup: 1.85
torch                                    mean time: 0.007588 ms
triton                                   mean time: 0.013859 ms, speedup: 0.55
tilelang                                 mean time: 0.014741 ms, speedup: 0.51
tilelang_vectorized                      mean time: 0.012294 ms, speedup: 0.62
tilelang_vectorized_eager                mean time: 0.007540 ms, speedup: 1.01
elementwise_add_fp16x8_packed            mean time: 0.007734 ms, speedup: 0.98
####################################################################################################
vector add, n: 4096
torch                                    mean time: 0.013263 ms
triton                                   mean time: 0.012376 ms, speedup: 1.07
tilelang                                 mean time: 0.007674 ms, speedup: 1.73
elementwise_add_fp32x4                   mean time: 0.012344 ms, speedup: 1.07
torch                                    mean time: 0.010063 ms
triton                                   mean time: 0.011412 ms, speedup: 0.88
tilelang                                 mean time: 0.011704 ms, speedup: 0.86
tilelang_vectorized                      mean time: 0.013151 ms, speedup: 0.77
tilelang_vectorized_eager                mean time: 0.008175 ms, speedup: 1.23
elementwise_add_fp16x8_packed            mean time: 0.007625 ms, speedup: 1.32
####################################################################################################
vector add, n: 32768
torch                                    mean time: 0.011004 ms
triton                                   mean time: 0.011101 ms, speedup: 0.99
tilelang                                 mean time: 0.007628 ms, speedup: 1.44
elementwise_add_fp32x4                   mean time: 0.013501 ms, speedup: 0.82
torch                                    mean time: 0.012603 ms
triton                                   mean time: 0.013592 ms, speedup: 0.93
tilelang                                 mean time: 0.007466 ms, speedup: 1.69
tilelang_vectorized                      mean time: 0.007771 ms, speedup: 1.62
tilelang_vectorized_eager                mean time: 0.008833 ms, speedup: 1.43
elementwise_add_fp16x8_packed            mean time: 0.015636 ms, speedup: 0.81
####################################################################################################
vector add, n: 1048576
torch                                    mean time: 0.011165 ms
triton                                   mean time: 0.014726 ms, speedup: 0.76
tilelang                                 mean time: 0.016001 ms, speedup: 0.70
elementwise_add_fp32x4                   mean time: 0.010641 ms, speedup: 1.05
torch                                    mean time: 0.008625 ms
triton                                   mean time: 0.013445 ms, speedup: 0.64
tilelang                                 mean time: 0.014476 ms, speedup: 0.60
tilelang_vectorized                      mean time: 0.010089 ms, speedup: 0.85
tilelang_vectorized_eager                mean time: 0.009659 ms, speedup: 0.89
elementwise_add_fp16x8_packed            mean time: 0.009290 ms, speedup: 0.93
####################################################################################################
vector add, n: 4194304
torch                                    mean time: 0.123677 ms
triton                                   mean time: 0.117502 ms, speedup: 1.05
tilelang                                 mean time: 0.117681 ms, speedup: 1.05
elementwise_add_fp32x4                   mean time: 0.118123 ms, speedup: 1.05
torch                                    mean time: 0.014838 ms
triton                                   mean time: 0.023633 ms, speedup: 0.63
tilelang                                 mean time: 0.036070 ms, speedup: 0.41
tilelang_vectorized                      mean time: 0.015490 ms, speedup: 0.96
tilelang_vectorized_eager                mean time: 0.025284 ms, speedup: 0.59
elementwise_add_fp16x8_packed            mean time: 0.014795 ms, speedup: 1.00
####################################################################################################
vector add, n: 16777216
torch                                    mean time: 0.594325 ms
triton                                   mean time: 0.592991 ms, speedup: 1.00
tilelang                                 mean time: 0.595282 ms, speedup: 1.00
elementwise_add_fp32x4                   mean time: 0.592963 ms, speedup: 1.00
torch                                    mean time: 0.297605 ms
triton                                   mean time: 0.298041 ms, speedup: 1.00
tilelang                                 mean time: 0.344466 ms, speedup: 0.86
tilelang_vectorized                      mean time: 0.297394 ms, speedup: 1.00
tilelang_vectorized_eager                mean time: 0.300754 ms, speedup: 0.99
elementwise_add_fp16x8_packed            mean time: 0.297142 ms, speedup: 1.00
```
