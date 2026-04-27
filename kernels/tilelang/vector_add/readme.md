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
torch                                    mean time: 0.003151 ms
triton                                   mean time: 0.009678 ms, speedup: 0.33
tilelang                                 mean time: 0.003366 ms, speedup: 0.94
####################################################################################################
vector add, n: 4096
torch                                    mean time: 0.003144 ms
triton                                   mean time: 0.009615 ms, speedup: 0.33
tilelang                                 mean time: 0.003341 ms, speedup: 0.94
####################################################################################################
vector add, n: 32768
torch                                    mean time: 0.003140 ms
triton                                   mean time: 0.009674 ms, speedup: 0.32
tilelang                                 mean time: 0.003328 ms, speedup: 0.94
####################################################################################################
vector add, n: 1048576
torch                                    mean time: 0.004799 ms
triton                                   mean time: 0.009657 ms, speedup: 0.50
tilelang                                 mean time: 0.004624 ms, speedup: 1.04
####################################################################################################
vector add, n: 4194304
torch                                    mean time: 0.012486 ms
triton                                   mean time: 0.010439 ms, speedup: 1.20
tilelang                                 mean time: 0.013003 ms, speedup: 0.96
####################################################################################################
vector add, n: 16777216
torch                                    mean time: 0.309612 ms
triton                                   mean time: 0.289336 ms, speedup: 1.07
tilelang                                 mean time: 0.284562 ms, speedup: 1.09
```
