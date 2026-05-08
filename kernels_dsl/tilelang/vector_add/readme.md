# triton_add

## Overview

- [x] triton add
- [x] tilelang add

## Run tests

```bash
python test.py
```

## Sample output

tested on L20

```yaml
####################################################################################################
vector add, n: 1024
torch                                    mean time: 0.008434 ms
triton                                   mean time: 0.010183 ms, speedup: 0.83
tilelang                                 mean time: 0.006893 ms, speedup: 1.22
elementwise_add_fp32x4                   mean time: 0.008146 ms, speedup: 1.04
torch                                    mean time: 0.007494 ms
triton                                   mean time: 0.010147 ms, speedup: 0.74
tilelang                                 mean time: 0.007551 ms, speedup: 0.99
elementwise_add_fp16x8_packed            mean time: 0.007206 ms, speedup: 1.04
####################################################################################################
vector add, n: 4096
torch                                    mean time: 0.011697 ms
triton                                   mean time: 0.014359 ms, speedup: 0.81
tilelang                                 mean time: 0.016012 ms, speedup: 0.73
elementwise_add_fp32x4                   mean time: 0.011481 ms, speedup: 1.02
torch                                    mean time: 0.007579 ms
triton                                   mean time: 0.009877 ms, speedup: 0.77
tilelang                                 mean time: 0.011529 ms, speedup: 0.66
elementwise_add_fp16x8_packed            mean time: 0.007735 ms, speedup: 0.98
####################################################################################################
vector add, n: 32768
torch                                    mean time: 0.008413 ms
triton                                   mean time: 0.014689 ms, speedup: 0.57
tilelang                                 mean time: 0.007829 ms, speedup: 1.07
elementwise_add_fp32x4                   mean time: 0.011658 ms, speedup: 0.72
torch                                    mean time: 0.008196 ms
triton                                   mean time: 0.011827 ms, speedup: 0.69
tilelang                                 mean time: 0.008293 ms, speedup: 0.99
elementwise_add_fp16x8_packed            mean time: 0.008290 ms, speedup: 0.99
####################################################################################################
vector add, n: 1048576
torch                                    mean time: 0.017082 ms
triton                                   mean time: 0.017316 ms, speedup: 0.99
tilelang                                 mean time: 0.026243 ms, speedup: 0.65
elementwise_add_fp32x4                   mean time: 0.013935 ms, speedup: 1.23
torch                                    mean time: 0.014598 ms
triton                                   mean time: 0.015936 ms, speedup: 0.92
tilelang                                 mean time: 0.024315 ms, speedup: 0.60
elementwise_add_fp16x8_packed            mean time: 0.011063 ms, speedup: 1.32
####################################################################################################
vector add, n: 4194304
torch                                    mean time: 0.134127 ms
triton                                   mean time: 0.130108 ms, speedup: 1.03
tilelang                                 mean time: 0.157289 ms, speedup: 0.85
elementwise_add_fp32x4                   mean time: 0.136387 ms, speedup: 0.98
torch                                    mean time: 0.022689 ms
triton                                   mean time: 0.035906 ms, speedup: 0.63
tilelang                                 mean time: 0.068169 ms, speedup: 0.33
elementwise_add_fp16x8_packed            mean time: 0.023165 ms, speedup: 0.98
####################################################################################################
vector add, n: 16777216
torch                                    mean time: 0.617635 ms
triton                                   mean time: 0.613414 ms, speedup: 1.01
tilelang                                 mean time: 0.614105 ms, speedup: 1.01
elementwise_add_fp32x4                   mean time: 0.612318 ms, speedup: 1.01
torch                                    mean time: 0.308373 ms
triton                                   mean time: 0.309220 ms, speedup: 1.00
tilelang                                 mean time: 0.346072 ms, speedup: 0.89
elementwise_add_fp16x8_packed            mean time: 0.308791 ms, speedup: 1.00
```
