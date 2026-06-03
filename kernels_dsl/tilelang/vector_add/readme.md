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
torch                                    mean time: 0.322368 ms
triton                                   mean time: 0.305424 ms, speedup: 1.06
tilelang                                 mean time: 0.370864 ms, speedup: 0.87
tilelang_vectorized                      mean time: 0.303072 ms, speedup: 1.06
tilelang_vectorized_eager                mean time: 0.303872 ms, speedup: 1.06
elementwise_add_fp16x8_packed            mean time: 0.304432 ms, speedup: 1.06
####################################################################################################
vector add, n: 16777217
torch                                    mean time: 0.304144 ms
triton                                   mean time: 0.304976 ms, speedup: 1.00
tilelang                                 mean time: 0.353232 ms, speedup: 0.86
tilelang_vectorized                      mean time: 0.303504 ms, speedup: 1.00
tilelang_vectorized_eager                mean time: 0.304832 ms, speedup: 1.00
elementwise_add_fp16x8_packed            mean time: 0.305072 ms, speedup: 1.00
```

### full benchmark

```yaml
####################################################################################################
vector add, n: 1024
torch                                    mean time: 0.007360 ms
triton                                   mean time: 0.007328 ms, speedup: 1.00
tilelang                                 mean time: 0.009632 ms, speedup: 0.76
elementwise_add_fp32x4                   mean time: 0.009568 ms, speedup: 0.77
torch                                    mean time: 0.007072 ms
triton                                   mean time: 0.006944 ms, speedup: 1.02
tilelang                                 mean time: 0.006992 ms, speedup: 1.01
tilelang_vectorized                      mean time: 0.006976 ms, speedup: 1.01
tilelang_vectorized_eager                mean time: 0.007024 ms, speedup: 1.01
elementwise_add_fp16x8_packed            mean time: 0.007120 ms, speedup: 0.99
####################################################################################################
vector add, n: 4096
torch                                    mean time: 0.007440 ms
triton                                   mean time: 0.007008 ms, speedup: 1.06
tilelang                                 mean time: 0.007040 ms, speedup: 1.06
elementwise_add_fp32x4                   mean time: 0.006944 ms, speedup: 1.07
torch                                    mean time: 0.006944 ms
triton                                   mean time: 0.007040 ms, speedup: 0.99
tilelang                                 mean time: 0.006944 ms, speedup: 1.00
tilelang_vectorized                      mean time: 0.007248 ms, speedup: 0.96
tilelang_vectorized_eager                mean time: 0.006944 ms, speedup: 1.00
elementwise_add_fp16x8_packed            mean time: 0.006944 ms, speedup: 1.00
####################################################################################################
vector add, n: 32768
torch                                    mean time: 0.007584 ms
triton                                   mean time: 0.007520 ms, speedup: 1.01
tilelang                                 mean time: 0.007616 ms, speedup: 1.00
elementwise_add_fp32x4                   mean time: 0.007440 ms, speedup: 1.02
torch                                    mean time: 0.007040 ms
triton                                   mean time: 0.007280 ms, speedup: 0.97
tilelang                                 mean time: 0.007264 ms, speedup: 0.97
tilelang_vectorized                      mean time: 0.007104 ms, speedup: 0.99
tilelang_vectorized_eager                mean time: 0.007408 ms, speedup: 0.95
elementwise_add_fp16x8_packed            mean time: 0.007136 ms, speedup: 0.99
####################################################################################################
vector add, n: 1048576
torch                                    mean time: 0.040304 ms
triton                                   mean time: 0.045808 ms, speedup: 0.88
tilelang                                 mean time: 0.045536 ms, speedup: 0.89
elementwise_add_fp32x4                   mean time: 0.046864 ms, speedup: 0.86
torch                                    mean time: 0.021648 ms
triton                                   mean time: 0.020240 ms, speedup: 1.07
tilelang                                 mean time: 0.034000 ms, speedup: 0.64
tilelang_vectorized                      mean time: 0.021632 ms, speedup: 1.00
tilelang_vectorized_eager                mean time: 0.021744 ms, speedup: 1.00
elementwise_add_fp16x8_packed            mean time: 0.021808 ms, speedup: 0.99
####################################################################################################
vector add, n: 4194304
torch                                    mean time: 0.156368 ms
triton                                   mean time: 0.160048 ms, speedup: 0.98
tilelang                                 mean time: 0.160784 ms, speedup: 0.97
elementwise_add_fp32x4                   mean time: 0.157584 ms, speedup: 0.99
torch                                    mean time: 0.080544 ms
triton                                   mean time: 0.080608 ms, speedup: 1.00
tilelang                                 mean time: 0.106656 ms, speedup: 0.76
tilelang_vectorized                      mean time: 0.077648 ms, speedup: 1.04
tilelang_vectorized_eager                mean time: 0.079216 ms, speedup: 1.02
elementwise_add_fp16x8_packed            mean time: 0.078336 ms, speedup: 1.03
####################################################################################################
vector add, n: 16777216
torch                                    mean time: 0.599504 ms
triton                                   mean time: 0.595184 ms, speedup: 1.01
tilelang                                 mean time: 0.599456 ms, speedup: 1.00
elementwise_add_fp32x4                   mean time: 0.593072 ms, speedup: 1.01
torch                                    mean time: 0.305136 ms
triton                                   mean time: 0.307120 ms, speedup: 0.99
tilelang                                 mean time: 0.358640 ms, speedup: 0.85
tilelang_vectorized                      mean time: 0.305072 ms, speedup: 1.00
tilelang_vectorized_eager                mean time: 0.304336 ms, speedup: 1.00
elementwise_add_fp16x8_packed            mean time: 0.304736 ms, speedup: 1.00
```
