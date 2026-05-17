# softmax

## Overview

- [x] tilelang softmax (naive two pass)

## Run tests

```bash
python test.py
```

## Sample output

### test_small

```yaml
#################################################################################################### test_small
bs: 256, hidden_size: 1024
torch                          mean time: 0.011951 ms
softmax_fp32x4                 mean time: 0.018472 ms, speedup: 0.65
tl.float32                     mean time: 0.023718 ms, speedup: 0.50
torch                          mean time: 0.015067 ms
softmax_arbitrary              mean time: 0.011484 ms, speedup: 1.31
tl.float16                     mean time: 0.015397 ms, speedup: 0.98
#################################################################################################### test_small
bs: 256, hidden_size: 4096
torch                          mean time: 0.021436 ms
softmax_fp32x4                 mean time: 0.009833 ms, speedup: 2.18
tl.float32                     mean time: 0.013116 ms, speedup: 1.63
torch                          mean time: 0.040158 ms
softmax_arbitrary              mean time: 0.015472 ms, speedup: 2.60
tl.float16                     mean time: 0.016545 ms, speedup: 2.43
#################################################################################################### test_small
bs: 256, hidden_size: 8192
torch                          mean time: 0.032023 ms
softmax_fp32x4                 mean time: 0.015456 ms, speedup: 2.07
tl.float32                     mean time: 0.021293 ms, speedup: 1.50
torch                          mean time: 0.029806 ms
softmax_arbitrary              mean time: 0.015809 ms, speedup: 1.89
tl.float16                     mean time: 0.019806 ms, speedup: 1.50
#################################################################################################### test_small
bs: 1024, hidden_size: 1024
torch                          mean time: 0.015480 ms
softmax_fp32x4                 mean time: 0.037665 ms, speedup: 0.41
tl.float32                     mean time: 0.013177 ms, speedup: 1.17
torch                          mean time: 0.013183 ms
softmax_arbitrary              mean time: 0.012257 ms, speedup: 1.08
tl.float16                     mean time: 0.014680 ms, speedup: 0.90
#################################################################################################### test_small
bs: 1024, hidden_size: 4096
torch                          mean time: 0.071040 ms
softmax_fp32x4                 mean time: 0.024029 ms, speedup: 2.96
tl.float32                     mean time: 0.049010 ms, speedup: 1.45
torch                          mean time: 0.070174 ms
softmax_arbitrary              mean time: 0.019279 ms, speedup: 3.64
tl.float16                     mean time: 0.025135 ms, speedup: 2.79
#################################################################################################### test_small
bs: 1024, hidden_size: 8192
torch                          mean time: 0.221538 ms
softmax_fp32x4                 mean time: 0.199897 ms, speedup: 1.11
tl.float32                     mean time: 0.190345 ms, speedup: 1.16
torch                          mean time: 0.145770 ms
softmax_arbitrary              mean time: 0.054861 ms, speedup: 2.66
tl.float16                     mean time: 0.071884 ms, speedup: 2.03
```

### test_large

```yaml
#################################################################################################### test_large
bs: 4, hidden_size: 16384
torch                          mean time: 0.008981 ms
softmax_medium                 mean time: 0.014834 ms, speedup: 0.61
softmax_extreme                mean time: 0.016692 ms, speedup: 0.54
softmax_arbitrary              mean time: 0.007295 ms, speedup: 1.23
softmax_splitk                 mean time: 0.017042 ms, speedup: 0.53
tl.float16                     mean time: 0.012269 ms, speedup: 0.73
#################################################################################################### test_large
bs: 4, hidden_size: 32768
torch                          mean time: 0.038188 ms
softmax_medium                 mean time: 0.009477 ms, speedup: 4.03
softmax_extreme                mean time: 0.013286 ms, speedup: 2.87
softmax_arbitrary              mean time: 0.011040 ms, speedup: 3.46
softmax_splitk                 mean time: 0.033487 ms, speedup: 1.14
tl.float16                     mean time: 0.024421 ms, speedup: 1.56
#################################################################################################### test_large
bs: 4, hidden_size: 65536
torch                          mean time: 0.014054 ms
softmax_extreme                mean time: 0.013864 ms, speedup: 1.01
softmax_arbitrary              mean time: 0.037239 ms, speedup: 0.38
softmax_splitk                 mean time: 0.029791 ms, speedup: 0.47
tl.float16                     mean time: 0.020349 ms, speedup: 0.69
#################################################################################################### test_large
bs: 4, hidden_size: 114688
torch                          mean time: 0.022157 ms
softmax_extreme                mean time: 0.027184 ms, speedup: 0.82
softmax_arbitrary              mean time: 0.020894 ms, speedup: 1.06
softmax_splitk                 mean time: 0.020024 ms, speedup: 1.11
tl.float16                     mean time: 0.030921 ms, speedup: 0.72
#################################################################################################### test_large
bs: 4, hidden_size: 262144
torch                          mean time: 0.040801 ms
softmax_arbitrary              mean time: 0.039732 ms, speedup: 1.03
softmax_splitk                 mean time: 0.021631 ms, speedup: 1.89
tl.float16                     mean time: 0.066521 ms, speedup: 0.61
#################################################################################################### test_large
bs: 4, hidden_size: 1048576
torch                          mean time: 0.145681 ms
softmax_arbitrary              mean time: 0.147920 ms, speedup: 0.98
softmax_splitk                 mean time: 0.024511 ms, speedup: 5.94
tl.float16                     mean time: 0.301371 ms, speedup: 0.48
#################################################################################################### test_large
bs: 4, hidden_size: 8388608
torch                          mean time: 2.195599 ms
softmax_arbitrary              mean time: 3.118947 ms, speedup: 0.70
softmax_splitk                 mean time: 0.659691 ms, speedup: 3.33
tl.float16                     mean time: 3.737053 ms, speedup: 0.59
#################################################################################################### test_large
bs: 4, hidden_size: 33554432
torch                          mean time: 12.502656 ms
softmax_arbitrary              mean time: 15.236486 ms, speedup: 0.82
softmax_splitk                 mean time: 2.770895 ms, speedup: 4.51
tl.float16                     mean time: 17.387890 ms, speedup: 0.72
```
