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
torch                                    mean time: 0.321152 ms
triton                                   mean time: 0.303328 ms, speedup: 1.06
tilelang                                 mean time: 0.374384 ms, speedup: 0.86
tilelang_vectorized                      mean time: 0.302240 ms, speedup: 1.06
tilelang_vectorized_eager                mean time: 0.314960 ms, speedup: 1.02
elementwise_add_fp16x8_packed            mean time: 0.304496 ms, speedup: 1.05
####################################################################################################
vector add, n: 16777217
torch                                    mean time: 0.302704 ms
triton                                   mean time: 0.302864 ms, speedup: 1.00
tilelang                                 mean time: 0.354928 ms, speedup: 0.85
tilelang_vectorized                      mean time: 0.301792 ms, speedup: 1.00
tilelang_vectorized_eager                mean time: 0.306816 ms, speedup: 0.99
elementwise_add_fp16x8_packed            mean time: 0.302768 ms, speedup: 1.00
```

### full benchmark

```yaml
####################################################################################################
vector add, n: 1024
torch                                    mean time: 0.007872 ms
triton                                   mean time: 0.008096 ms, speedup: 0.97
tilelang                                 mean time: 0.007040 ms, speedup: 1.12
elementwise_add_fp32x4                   mean time: 0.007136 ms, speedup: 1.10
torch                                    mean time: 0.007008 ms
triton                                   mean time: 0.006976 ms, speedup: 1.00
tilelang                                 mean time: 0.007040 ms, speedup: 1.00
tilelang_vectorized                      mean time: 0.007104 ms, speedup: 0.99
tilelang_vectorized_eager                mean time: 0.007296 ms, speedup: 0.96
elementwise_add_fp16x8_packed            mean time: 0.007088 ms, speedup: 0.99
####################################################################################################
vector add, n: 4096
torch                                    mean time: 0.007040 ms
triton                                   mean time: 0.006976 ms, speedup: 1.01
tilelang                                 mean time: 0.007168 ms, speedup: 0.98
elementwise_add_fp32x4                   mean time: 0.007024 ms, speedup: 1.00
torch                                    mean time: 0.007104 ms
triton                                   mean time: 0.007232 ms, speedup: 0.98
tilelang                                 mean time: 0.007136 ms, speedup: 1.00
tilelang_vectorized                      mean time: 0.006976 ms, speedup: 1.02
tilelang_vectorized_eager                mean time: 0.007376 ms, speedup: 0.96
elementwise_add_fp16x8_packed            mean time: 0.007008 ms, speedup: 1.01
####################################################################################################
vector add, n: 32768
torch                                    mean time: 0.007856 ms
triton                                   mean time: 0.008000 ms, speedup: 0.98
tilelang                                 mean time: 0.007568 ms, speedup: 1.04
elementwise_add_fp32x4                   mean time: 0.008640 ms, speedup: 0.91
torch                                    mean time: 0.007440 ms
triton                                   mean time: 0.007456 ms, speedup: 1.00
tilelang                                 mean time: 0.007344 ms, speedup: 1.01
tilelang_vectorized                      mean time: 0.007376 ms, speedup: 1.01
tilelang_vectorized_eager                mean time: 0.007360 ms, speedup: 1.01
elementwise_add_fp16x8_packed            mean time: 0.007344 ms, speedup: 1.01
####################################################################################################
vector add, n: 1048576
torch                                    mean time: 0.039936 ms
triton                                   mean time: 0.038528 ms, speedup: 1.04
tilelang                                 mean time: 0.039168 ms, speedup: 1.02
elementwise_add_fp32x4                   mean time: 0.040272 ms, speedup: 0.99
torch                                    mean time: 0.022176 ms
triton                                   mean time: 0.020288 ms, speedup: 1.09
tilelang                                 mean time: 0.033984 ms, speedup: 0.65
tilelang_vectorized                      mean time: 0.023088 ms, speedup: 0.96
tilelang_vectorized_eager                mean time: 0.028192 ms, speedup: 0.79
elementwise_add_fp16x8_packed            mean time: 0.028384 ms, speedup: 0.78
####################################################################################################
vector add, n: 4194304
torch                                    mean time: 0.156192 ms
triton                                   mean time: 0.159392 ms, speedup: 0.98
tilelang                                 mean time: 0.166112 ms, speedup: 0.94
elementwise_add_fp32x4                   mean time: 0.167312 ms, speedup: 0.93
torch                                    mean time: 0.081952 ms
triton                                   mean time: 0.085488 ms, speedup: 0.96
tilelang                                 mean time: 0.113456 ms, speedup: 0.72
tilelang_vectorized                      mean time: 0.083056 ms, speedup: 0.99
tilelang_vectorized_eager                mean time: 0.081600 ms, speedup: 1.00
elementwise_add_fp16x8_packed            mean time: 0.081344 ms, speedup: 1.01
####################################################################################################
vector add, n: 16777216
torch                                    mean time: 0.593824 ms
triton                                   mean time: 0.589920 ms, speedup: 1.01
tilelang                                 mean time: 0.597568 ms, speedup: 0.99
elementwise_add_fp32x4                   mean time: 0.589872 ms, speedup: 1.01
torch                                    mean time: 0.300944 ms
triton                                   mean time: 0.309360 ms, speedup: 0.97
tilelang                                 mean time: 0.353776 ms, speedup: 0.85
tilelang_vectorized                      mean time: 0.303424 ms, speedup: 0.99
tilelang_vectorized_eager                mean time: 0.312832 ms, speedup: 0.96
elementwise_add_fp16x8_packed            mean time: 0.301520 ms, speedup: 1.00
```
