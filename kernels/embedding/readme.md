# embedding

## Overview

Embedding lookup kernels.

- [x] embedding — FP32 / FP16
- [x] embedding_fp32x4 — vectorized FP32
- [x] embedding_fp32x4_packed — vectorized FP32, packed r/w
- [x] embedding_fp16x2 — vectorized FP16
- [x] embedding_fp16x8 — vectorized FP16
- [x] embedding_fp16x8_packed — vectorized FP16, packed r/w
- [x] pytorch op bindings && diff check

## Run tests

Large embedding tables need a GPU with plenty of VRAM.

```bash
export TORCH_CUDA_ARCH_LIST=8.9
python test.py
```

### Sample output

```bash
####################################################################################################
emd_size: 32, emd_dim: 32, seq_len: 1
torch                          mean time: 0.009232 ms
embedding                      mean time: 0.007888 ms, speedup: 1.17
embedding_fp32x4               mean time: 0.007040 ms, speedup: 1.31
embedding_fp32x4_packed        mean time: 0.007872 ms, speedup: 1.17
torch                          mean time: 0.007152 ms
embedding_half                 mean time: 0.007776 ms, speedup: 0.92
embedding_fp16x8               mean time: 0.007296 ms, speedup: 0.98
embedding_fp16x8_packed        mean time: 0.007024 ms, speedup: 1.02
####################################################################################################
emd_size: 32, emd_dim: 32, seq_len: 64
torch                          mean time: 0.007136 ms
embedding                      mean time: 0.007104 ms, speedup: 1.00
embedding_fp32x4               mean time: 0.007456 ms, speedup: 0.96
embedding_fp32x4_packed        mean time: 0.007152 ms, speedup: 1.00
torch                          mean time: 0.007104 ms
embedding_half                 mean time: 0.007136 ms, speedup: 1.00
embedding_fp16x8               mean time: 0.007152 ms, speedup: 0.99
embedding_fp16x8_packed        mean time: 0.007120 ms, speedup: 1.00
####################################################################################################
emd_size: 32, emd_dim: 32, seq_len: 256
torch                          mean time: 0.007824 ms
embedding                      mean time: 0.007328 ms, speedup: 1.07
embedding_fp32x4               mean time: 0.007152 ms, speedup: 1.09
embedding_fp32x4_packed        mean time: 0.007200 ms, speedup: 1.09
torch                          mean time: 0.007136 ms
embedding_half                 mean time: 0.007184 ms, speedup: 0.99
embedding_fp16x8               mean time: 0.007152 ms, speedup: 1.00
embedding_fp16x8_packed        mean time: 0.007008 ms, speedup: 1.02
####################################################################################################
emd_size: 32, emd_dim: 32, seq_len: 1024
torch                          mean time: 0.008784 ms
embedding                      mean time: 0.009136 ms, speedup: 0.96
embedding_fp32x4               mean time: 0.008464 ms, speedup: 1.04
embedding_fp32x4_packed        mean time: 0.007680 ms, speedup: 1.14
torch                          mean time: 0.008464 ms
embedding_half                 mean time: 0.007840 ms, speedup: 1.08
embedding_fp16x8               mean time: 0.008480 ms, speedup: 1.00
embedding_fp16x8_packed        mean time: 0.007760 ms, speedup: 1.09
####################################################################################################
emd_size: 32, emd_dim: 128, seq_len: 1
torch                          mean time: 0.007296 ms
embedding                      mean time: 0.007632 ms, speedup: 0.96
embedding_fp32x4               mean time: 0.007088 ms, speedup: 1.03
embedding_fp32x4_packed        mean time: 0.006944 ms, speedup: 1.05
torch                          mean time: 0.007152 ms
embedding_half                 mean time: 0.007120 ms, speedup: 1.00
embedding_fp16x8               mean time: 0.007408 ms, speedup: 0.97
embedding_fp16x8_packed        mean time: 0.007008 ms, speedup: 1.02
####################################################################################################
emd_size: 32, emd_dim: 128, seq_len: 64
torch                          mean time: 0.007328 ms
embedding                      mean time: 0.007136 ms, speedup: 1.03
embedding_fp32x4               mean time: 0.007232 ms, speedup: 1.01
embedding_fp32x4_packed        mean time: 0.007184 ms, speedup: 1.02
torch                          mean time: 0.007120 ms
embedding_half                 mean time: 0.007200 ms, speedup: 0.99
embedding_fp16x8               mean time: 0.007200 ms, speedup: 0.99
embedding_fp16x8_packed        mean time: 0.007200 ms, speedup: 0.99
####################################################################################################
emd_size: 32, emd_dim: 128, seq_len: 256
torch                          mean time: 0.007232 ms
embedding                      mean time: 0.007072 ms, speedup: 1.02
embedding_fp32x4               mean time: 0.007248 ms, speedup: 1.00
embedding_fp32x4_packed        mean time: 0.007312 ms, speedup: 0.99
torch                          mean time: 0.007136 ms
embedding_half                 mean time: 0.007168 ms, speedup: 1.00
embedding_fp16x8               mean time: 0.007648 ms, speedup: 0.93
embedding_fp16x8_packed        mean time: 0.007360 ms, speedup: 0.97
####################################################################################################
emd_size: 32, emd_dim: 128, seq_len: 1024
torch                          mean time: 0.008544 ms
embedding                      mean time: 0.009056 ms, speedup: 0.94
embedding_fp32x4               mean time: 0.009056 ms, speedup: 0.94
embedding_fp32x4_packed        mean time: 0.007680 ms, speedup: 1.11
torch                          mean time: 0.007792 ms
embedding_half                 mean time: 0.009424 ms, speedup: 0.83
embedding_fp16x8               mean time: 0.009104 ms, speedup: 0.86
embedding_fp16x8_packed        mean time: 0.007600 ms, speedup: 1.03
####################################################################################################
emd_size: 32, emd_dim: 512, seq_len: 1
torch                          mean time: 0.007856 ms
embedding                      mean time: 0.006960 ms, speedup: 1.13
embedding_fp32x4               mean time: 0.007152 ms, speedup: 1.10
embedding_fp32x4_packed        mean time: 0.007152 ms, speedup: 1.10
torch                          mean time: 0.007520 ms
embedding_half                 mean time: 0.007568 ms, speedup: 0.99
embedding_fp16x8               mean time: 0.007584 ms, speedup: 0.99
embedding_fp16x8_packed        mean time: 0.007024 ms, speedup: 1.07
####################################################################################################
emd_size: 32, emd_dim: 512, seq_len: 64
torch                          mean time: 0.007216 ms
embedding                      mean time: 0.007632 ms, speedup: 0.95
embedding_fp32x4               mean time: 0.007312 ms, speedup: 0.99
embedding_fp32x4_packed        mean time: 0.007264 ms, speedup: 0.99
torch                          mean time: 0.007520 ms
embedding_half                 mean time: 0.007216 ms, speedup: 1.04
embedding_fp16x8               mean time: 0.007488 ms, speedup: 1.00
embedding_fp16x8_packed        mean time: 0.007104 ms, speedup: 1.06
####################################################################################################
emd_size: 32, emd_dim: 512, seq_len: 256
torch                          mean time: 0.007184 ms
embedding                      mean time: 0.009344 ms, speedup: 0.77
embedding_fp32x4               mean time: 0.009024 ms, speedup: 0.80
embedding_fp32x4_packed        mean time: 0.007664 ms, speedup: 0.94
torch                          mean time: 0.007264 ms
embedding_half                 mean time: 0.009296 ms, speedup: 0.78
embedding_fp16x8               mean time: 0.009184 ms, speedup: 0.79
embedding_fp16x8_packed        mean time: 0.007200 ms, speedup: 1.01
####################################################################################################
emd_size: 32, emd_dim: 512, seq_len: 1024
torch                          mean time: 0.011168 ms
embedding                      mean time: 0.016992 ms, speedup: 0.66
embedding_fp32x4               mean time: 0.014848 ms, speedup: 0.75
embedding_fp32x4_packed        mean time: 0.010608 ms, speedup: 1.05
torch                          mean time: 0.008896 ms
embedding_half                 mean time: 0.015760 ms, speedup: 0.56
embedding_fp16x8               mean time: 0.017424 ms, speedup: 0.51
embedding_fp16x8_packed        mean time: 0.008864 ms, speedup: 1.00
####################################################################################################
emd_size: 128, emd_dim: 32, seq_len: 1
torch                          mean time: 0.007360 ms
embedding                      mean time: 0.007072 ms, speedup: 1.04
embedding_fp32x4               mean time: 0.007024 ms, speedup: 1.05
embedding_fp32x4_packed        mean time: 0.007024 ms, speedup: 1.05
torch                          mean time: 0.007280 ms
embedding_half                 mean time: 0.007072 ms, speedup: 1.03
embedding_fp16x8               mean time: 0.007104 ms, speedup: 1.02
embedding_fp16x8_packed        mean time: 0.007456 ms, speedup: 0.98
####################################################################################################
emd_size: 128, emd_dim: 32, seq_len: 64
torch                          mean time: 0.008048 ms
embedding                      mean time: 0.007248 ms, speedup: 1.11
embedding_fp32x4               mean time: 0.007136 ms, speedup: 1.13
embedding_fp32x4_packed        mean time: 0.007328 ms, speedup: 1.10
torch                          mean time: 0.007536 ms
embedding_half                 mean time: 0.007184 ms, speedup: 1.05
embedding_fp16x8               mean time: 0.007520 ms, speedup: 1.00
embedding_fp16x8_packed        mean time: 0.007184 ms, speedup: 1.05
####################################################################################################
emd_size: 128, emd_dim: 32, seq_len: 256
torch                          mean time: 0.007568 ms
embedding                      mean time: 0.007264 ms, speedup: 1.04
embedding_fp32x4               mean time: 0.007344 ms, speedup: 1.03
embedding_fp32x4_packed        mean time: 0.007296 ms, speedup: 1.04
torch                          mean time: 0.007744 ms
embedding_half                 mean time: 0.007392 ms, speedup: 1.05
embedding_fp16x8               mean time: 0.007376 ms, speedup: 1.05
embedding_fp16x8_packed        mean time: 0.007232 ms, speedup: 1.07
####################################################################################################
emd_size: 128, emd_dim: 32, seq_len: 1024
torch                          mean time: 0.008992 ms
embedding                      mean time: 0.008864 ms, speedup: 1.01
embedding_fp32x4               mean time: 0.009024 ms, speedup: 1.00
embedding_fp32x4_packed        mean time: 0.008896 ms, speedup: 1.01
torch                          mean time: 0.008896 ms
embedding_half                 mean time: 0.008672 ms, speedup: 1.03
embedding_fp16x8               mean time: 0.009216 ms, speedup: 0.97
embedding_fp16x8_packed        mean time: 0.008896 ms, speedup: 1.00
####################################################################################################
emd_size: 128, emd_dim: 128, seq_len: 1
torch                          mean time: 0.007520 ms
embedding                      mean time: 0.007312 ms, speedup: 1.03
embedding_fp32x4               mean time: 0.007136 ms, speedup: 1.05
embedding_fp32x4_packed        mean time: 0.007472 ms, speedup: 1.01
torch                          mean time: 0.007648 ms
embedding_half                 mean time: 0.007072 ms, speedup: 1.08
embedding_fp16x8               mean time: 0.007168 ms, speedup: 1.07
embedding_fp16x8_packed        mean time: 0.007312 ms, speedup: 1.05
####################################################################################################
emd_size: 128, emd_dim: 128, seq_len: 64
torch                          mean time: 0.007232 ms
embedding                      mean time: 0.007360 ms, speedup: 0.98
embedding_fp32x4               mean time: 0.007424 ms, speedup: 0.97
embedding_fp32x4_packed        mean time: 0.007248 ms, speedup: 1.00
torch                          mean time: 0.007296 ms
embedding_half                 mean time: 0.007152 ms, speedup: 1.02
embedding_fp16x8               mean time: 0.007200 ms, speedup: 1.01
embedding_fp16x8_packed        mean time: 0.007872 ms, speedup: 0.93
####################################################################################################
emd_size: 128, emd_dim: 128, seq_len: 256
torch                          mean time: 0.008016 ms
embedding                      mean time: 0.007488 ms, speedup: 1.07
embedding_fp32x4               mean time: 0.007360 ms, speedup: 1.09
embedding_fp32x4_packed        mean time: 0.007456 ms, speedup: 1.08
torch                          mean time: 0.007280 ms
embedding_half                 mean time: 0.007360 ms, speedup: 0.99
embedding_fp16x8               mean time: 0.007456 ms, speedup: 0.98
embedding_fp16x8_packed        mean time: 0.007376 ms, speedup: 0.99
####################################################################################################
emd_size: 128, emd_dim: 128, seq_len: 1024
torch                          mean time: 0.008864 ms
embedding                      mean time: 0.009376 ms, speedup: 0.95
embedding_fp32x4               mean time: 0.009360 ms, speedup: 0.95
embedding_fp32x4_packed        mean time: 0.008960 ms, speedup: 0.99
torch                          mean time: 0.008848 ms
embedding_half                 mean time: 0.009888 ms, speedup: 0.89
embedding_fp16x8               mean time: 0.010048 ms, speedup: 0.88
embedding_fp16x8_packed        mean time: 0.008928 ms, speedup: 0.99
####################################################################################################
emd_size: 128, emd_dim: 512, seq_len: 1
torch                          mean time: 0.007632 ms
embedding                      mean time: 0.007168 ms, speedup: 1.06
embedding_fp32x4               mean time: 0.007536 ms, speedup: 1.01
embedding_fp32x4_packed        mean time: 0.007232 ms, speedup: 1.06
torch                          mean time: 0.007584 ms
embedding_half                 mean time: 0.007984 ms, speedup: 0.95
embedding_fp16x8               mean time: 0.007232 ms, speedup: 1.05
embedding_fp16x8_packed        mean time: 0.007072 ms, speedup: 1.07
####################################################################################################
emd_size: 128, emd_dim: 512, seq_len: 64
torch                          mean time: 0.008288 ms
embedding                      mean time: 0.007984 ms, speedup: 1.04
embedding_fp32x4               mean time: 0.008016 ms, speedup: 1.03
embedding_fp32x4_packed        mean time: 0.008800 ms, speedup: 0.94
torch                          mean time: 0.007456 ms
embedding_half                 mean time: 0.007296 ms, speedup: 1.02
embedding_fp16x8               mean time: 0.007968 ms, speedup: 0.94
embedding_fp16x8_packed        mean time: 0.007552 ms, speedup: 0.99
####################################################################################################
emd_size: 128, emd_dim: 512, seq_len: 256
torch                          mean time: 0.008896 ms
embedding                      mean time: 0.011200 ms, speedup: 0.79
embedding_fp32x4               mean time: 0.009216 ms, speedup: 0.97
embedding_fp32x4_packed        mean time: 0.008896 ms, speedup: 1.00
torch                          mean time: 0.008032 ms
embedding_half                 mean time: 0.009712 ms, speedup: 0.83
embedding_fp16x8               mean time: 0.009456 ms, speedup: 0.85
embedding_fp16x8_packed        mean time: 0.007840 ms, speedup: 1.02
####################################################################################################
emd_size: 128, emd_dim: 512, seq_len: 1024
torch                          mean time: 0.011056 ms
embedding                      mean time: 0.018848 ms, speedup: 0.59
embedding_fp32x4               mean time: 0.017168 ms, speedup: 0.64
embedding_fp32x4_packed        mean time: 0.011232 ms, speedup: 0.98
torch                          mean time: 0.009184 ms
embedding_half                 mean time: 0.017648 ms, speedup: 0.52
embedding_fp16x8               mean time: 0.019248 ms, speedup: 0.48
embedding_fp16x8_packed        mean time: 0.009120 ms, speedup: 1.01
####################################################################################################
emd_size: 1024, emd_dim: 32, seq_len: 1
torch                          mean time: 0.007520 ms
embedding                      mean time: 0.007568 ms, speedup: 0.99
embedding_fp32x4               mean time: 0.007040 ms, speedup: 1.07
embedding_fp32x4_packed        mean time: 0.007664 ms, speedup: 0.98
torch                          mean time: 0.007520 ms
embedding_half                 mean time: 0.007136 ms, speedup: 1.05
embedding_fp16x8               mean time: 0.007424 ms, speedup: 1.01
embedding_fp16x8_packed        mean time: 0.007232 ms, speedup: 1.04
####################################################################################################
emd_size: 1024, emd_dim: 32, seq_len: 64
torch                          mean time: 0.007744 ms
embedding                      mean time: 0.007216 ms, speedup: 1.07
embedding_fp32x4               mean time: 0.007296 ms, speedup: 1.06
embedding_fp32x4_packed        mean time: 0.007200 ms, speedup: 1.08
torch                          mean time: 0.007200 ms
embedding_half                 mean time: 0.007264 ms, speedup: 0.99
embedding_fp16x8               mean time: 0.007296 ms, speedup: 0.99
embedding_fp16x8_packed        mean time: 0.007168 ms, speedup: 1.00
####################################################################################################
emd_size: 1024, emd_dim: 32, seq_len: 256
torch                          mean time: 0.007632 ms
embedding                      mean time: 0.007200 ms, speedup: 1.06
embedding_fp32x4               mean time: 0.007296 ms, speedup: 1.05
embedding_fp32x4_packed        mean time: 0.007280 ms, speedup: 1.05
torch                          mean time: 0.007520 ms
embedding_half                 mean time: 0.007248 ms, speedup: 1.04
embedding_fp16x8               mean time: 0.007824 ms, speedup: 0.96
embedding_fp16x8_packed        mean time: 0.007232 ms, speedup: 1.04
####################################################################################################
emd_size: 1024, emd_dim: 32, seq_len: 1024
torch                          mean time: 0.008992 ms
embedding                      mean time: 0.008992 ms, speedup: 1.00
embedding_fp32x4               mean time: 0.009152 ms, speedup: 0.98
embedding_fp32x4_packed        mean time: 0.009056 ms, speedup: 0.99
torch                          mean time: 0.009040 ms
embedding_half                 mean time: 0.009088 ms, speedup: 0.99
embedding_fp16x8               mean time: 0.009088 ms, speedup: 0.99
embedding_fp16x8_packed        mean time: 0.008864 ms, speedup: 1.02
####################################################################################################
emd_size: 1024, emd_dim: 128, seq_len: 1
torch                          mean time: 0.007984 ms
embedding                      mean time: 0.007264 ms, speedup: 1.10
embedding_fp32x4               mean time: 0.007360 ms, speedup: 1.08
embedding_fp32x4_packed        mean time: 0.007328 ms, speedup: 1.09
torch                          mean time: 0.007552 ms
embedding_half                 mean time: 0.007136 ms, speedup: 1.06
embedding_fp16x8               mean time: 0.007216 ms, speedup: 1.05
embedding_fp16x8_packed        mean time: 0.007520 ms, speedup: 1.00
####################################################################################################
emd_size: 1024, emd_dim: 128, seq_len: 64
torch                          mean time: 0.007248 ms
embedding                      mean time: 0.007232 ms, speedup: 1.00
embedding_fp32x4               mean time: 0.007472 ms, speedup: 0.97
embedding_fp32x4_packed        mean time: 0.007376 ms, speedup: 0.98
torch                          mean time: 0.007168 ms
embedding_half                 mean time: 0.007104 ms, speedup: 1.01
embedding_fp16x8               mean time: 0.007328 ms, speedup: 0.98
embedding_fp16x8_packed        mean time: 0.007280 ms, speedup: 0.98
####################################################################################################
emd_size: 1024, emd_dim: 128, seq_len: 256
torch                          mean time: 0.007456 ms
embedding                      mean time: 0.007504 ms, speedup: 0.99
embedding_fp32x4               mean time: 0.008800 ms, speedup: 0.85
embedding_fp32x4_packed        mean time: 0.007520 ms, speedup: 0.99
torch                          mean time: 0.007520 ms
embedding_half                 mean time: 0.007392 ms, speedup: 1.02
embedding_fp16x8               mean time: 0.007488 ms, speedup: 1.00
embedding_fp16x8_packed        mean time: 0.007872 ms, speedup: 0.96
####################################################################################################
emd_size: 1024, emd_dim: 128, seq_len: 1024
torch                          mean time: 0.009376 ms
embedding                      mean time: 0.011168 ms, speedup: 0.84
embedding_fp32x4               mean time: 0.009536 ms, speedup: 0.98
embedding_fp32x4_packed        mean time: 0.009456 ms, speedup: 0.99
torch                          mean time: 0.009072 ms
embedding_half                 mean time: 0.010304 ms, speedup: 0.88
embedding_fp16x8               mean time: 0.009760 ms, speedup: 0.93
embedding_fp16x8_packed        mean time: 0.009280 ms, speedup: 0.98
####################################################################################################
emd_size: 1024, emd_dim: 512, seq_len: 1
torch                          mean time: 0.007728 ms
embedding                      mean time: 0.007296 ms, speedup: 1.06
embedding_fp32x4               mean time: 0.007296 ms, speedup: 1.06
embedding_fp32x4_packed        mean time: 0.007296 ms, speedup: 1.06
torch                          mean time: 0.007920 ms
embedding_half                 mean time: 0.007424 ms, speedup: 1.07
embedding_fp16x8               mean time: 0.007232 ms, speedup: 1.10
embedding_fp16x8_packed        mean time: 0.007408 ms, speedup: 1.07
####################################################################################################
emd_size: 1024, emd_dim: 512, seq_len: 64
torch                          mean time: 0.008912 ms
embedding                      mean time: 0.008800 ms, speedup: 1.01
embedding_fp32x4               mean time: 0.008848 ms, speedup: 1.01
embedding_fp32x4_packed        mean time: 0.009104 ms, speedup: 0.98
torch                          mean time: 0.007536 ms
embedding_half                 mean time: 0.007600 ms, speedup: 0.99
embedding_fp16x8               mean time: 0.009168 ms, speedup: 0.82
embedding_fp16x8_packed        mean time: 0.007648 ms, speedup: 0.99
####################################################################################################
emd_size: 1024, emd_dim: 512, seq_len: 256
torch                          mean time: 0.009264 ms
embedding                      mean time: 0.011344 ms, speedup: 0.82
embedding_fp32x4               mean time: 0.009376 ms, speedup: 0.99
embedding_fp32x4_packed        mean time: 0.009296 ms, speedup: 1.00
torch                          mean time: 0.009104 ms
embedding_half                 mean time: 0.011184 ms, speedup: 0.81
embedding_fp16x8               mean time: 0.009536 ms, speedup: 0.95
embedding_fp16x8_packed        mean time: 0.008992 ms, speedup: 1.01
####################################################################################################
emd_size: 1024, emd_dim: 512, seq_len: 1024
torch                          mean time: 0.013872 ms
embedding                      mean time: 0.025040 ms, speedup: 0.55
embedding_fp32x4               mean time: 0.017616 ms, speedup: 0.79
embedding_fp32x4_packed        mean time: 0.013712 ms, speedup: 1.01
torch                          mean time: 0.011616 ms
embedding_half                 mean time: 0.021584 ms, speedup: 0.54
embedding_fp16x8               mean time: 0.019424 ms, speedup: 0.60
embedding_fp16x8_packed        mean time: 0.011216 ms, speedup: 1.04
####################################################################################################
emd_size: 102400, emd_dim: 32, seq_len: 1
torch                          mean time: 0.007536 ms
embedding                      mean time: 0.007136 ms, speedup: 1.06
embedding_fp32x4               mean time: 0.007136 ms, speedup: 1.06
embedding_fp32x4_packed        mean time: 0.007056 ms, speedup: 1.07
torch                          mean time: 0.007712 ms
embedding_half                 mean time: 0.007104 ms, speedup: 1.09
embedding_fp16x8               mean time: 0.007104 ms, speedup: 1.09
embedding_fp16x8_packed        mean time: 0.007040 ms, speedup: 1.10
####################################################################################################
emd_size: 102400, emd_dim: 32, seq_len: 64
torch                          mean time: 0.007168 ms
embedding                      mean time: 0.007296 ms, speedup: 0.98
embedding_fp32x4               mean time: 0.007280 ms, speedup: 0.98
embedding_fp32x4_packed        mean time: 0.007072 ms, speedup: 1.01
torch                          mean time: 0.007232 ms
embedding_half                 mean time: 0.007584 ms, speedup: 0.95
embedding_fp16x8               mean time: 0.007424 ms, speedup: 0.97
embedding_fp16x8_packed        mean time: 0.007392 ms, speedup: 0.98
####################################################################################################
emd_size: 102400, emd_dim: 32, seq_len: 256
torch                          mean time: 0.007520 ms
embedding                      mean time: 0.007280 ms, speedup: 1.03
embedding_fp32x4               mean time: 0.007392 ms, speedup: 1.02
embedding_fp32x4_packed        mean time: 0.007360 ms, speedup: 1.02
torch                          mean time: 0.007392 ms
embedding_half                 mean time: 0.007312 ms, speedup: 1.01
embedding_fp16x8               mean time: 0.007360 ms, speedup: 1.00
embedding_fp16x8_packed        mean time: 0.007264 ms, speedup: 1.02
####################################################################################################
emd_size: 102400, emd_dim: 32, seq_len: 1024
torch                          mean time: 0.009216 ms
embedding                      mean time: 0.009136 ms, speedup: 1.01
embedding_fp32x4               mean time: 0.009184 ms, speedup: 1.00
embedding_fp32x4_packed        mean time: 0.009184 ms, speedup: 1.00
torch                          mean time: 0.009152 ms
embedding_half                 mean time: 0.009120 ms, speedup: 1.00
embedding_fp16x8               mean time: 0.009200 ms, speedup: 0.99
embedding_fp16x8_packed        mean time: 0.009184 ms, speedup: 1.00
####################################################################################################
emd_size: 102400, emd_dim: 128, seq_len: 1
torch                          mean time: 0.007664 ms
embedding                      mean time: 0.008048 ms, speedup: 0.95
embedding_fp32x4               mean time: 0.007472 ms, speedup: 1.03
embedding_fp32x4_packed        mean time: 0.007184 ms, speedup: 1.07
torch                          mean time: 0.007952 ms
embedding_half                 mean time: 0.007296 ms, speedup: 1.09
embedding_fp16x8               mean time: 0.007520 ms, speedup: 1.06
embedding_fp16x8_packed        mean time: 0.007168 ms, speedup: 1.11
####################################################################################################
emd_size: 102400, emd_dim: 128, seq_len: 64
torch                          mean time: 0.007232 ms
embedding                      mean time: 0.007248 ms, speedup: 1.00
embedding_fp32x4               mean time: 0.007872 ms, speedup: 0.92
embedding_fp32x4_packed        mean time: 0.007312 ms, speedup: 0.99
torch                          mean time: 0.007328 ms
embedding_half                 mean time: 0.007232 ms, speedup: 1.01
embedding_fp16x8               mean time: 0.007296 ms, speedup: 1.00
embedding_fp16x8_packed        mean time: 0.007232 ms, speedup: 1.01
####################################################################################################
emd_size: 102400, emd_dim: 128, seq_len: 256
torch                          mean time: 0.008576 ms
embedding                      mean time: 0.008640 ms, speedup: 0.99
embedding_fp32x4               mean time: 0.008928 ms, speedup: 0.96
embedding_fp32x4_packed        mean time: 0.008048 ms, speedup: 1.07
torch                          mean time: 0.007472 ms
embedding_half                 mean time: 0.008448 ms, speedup: 0.88
embedding_fp16x8               mean time: 0.007552 ms, speedup: 0.99
embedding_fp16x8_packed        mean time: 0.007424 ms, speedup: 1.01
####################################################################################################
emd_size: 102400, emd_dim: 128, seq_len: 1024
torch                          mean time: 0.011104 ms
embedding                      mean time: 0.011456 ms, speedup: 0.97
embedding_fp32x4               mean time: 0.011008 ms, speedup: 1.01
embedding_fp32x4_packed        mean time: 0.011264 ms, speedup: 0.99
torch                          mean time: 0.009344 ms
embedding_half                 mean time: 0.011136 ms, speedup: 0.84
embedding_fp16x8               mean time: 0.011104 ms, speedup: 0.84
embedding_fp16x8_packed        mean time: 0.009424 ms, speedup: 0.99
####################################################################################################
emd_size: 102400, emd_dim: 512, seq_len: 1
torch                          mean time: 0.007600 ms
embedding                      mean time: 0.007200 ms, speedup: 1.06
embedding_fp32x4               mean time: 0.007152 ms, speedup: 1.06
embedding_fp32x4_packed        mean time: 0.007568 ms, speedup: 1.00
torch                          mean time: 0.007776 ms
embedding_half                 mean time: 0.007856 ms, speedup: 0.99
embedding_fp16x8               mean time: 0.007424 ms, speedup: 1.05
embedding_fp16x8_packed        mean time: 0.007264 ms, speedup: 1.07
####################################################################################################
emd_size: 102400, emd_dim: 512, seq_len: 64
torch                          mean time: 0.007840 ms
embedding                      mean time: 0.007680 ms, speedup: 1.02
embedding_fp32x4               mean time: 0.009040 ms, speedup: 0.87
embedding_fp32x4_packed        mean time: 0.008416 ms, speedup: 0.93
torch                          mean time: 0.007456 ms
embedding_half                 mean time: 0.007440 ms, speedup: 1.00
embedding_fp16x8               mean time: 0.007744 ms, speedup: 0.96
embedding_fp16x8_packed        mean time: 0.007520 ms, speedup: 0.99
####################################################################################################
emd_size: 102400, emd_dim: 512, seq_len: 256
torch                          mean time: 0.010992 ms
embedding                      mean time: 0.011888 ms, speedup: 0.92
embedding_fp32x4               mean time: 0.009808 ms, speedup: 1.12
embedding_fp32x4_packed        mean time: 0.009600 ms, speedup: 1.14
torch                          mean time: 0.009248 ms
embedding_half                 mean time: 0.011136 ms, speedup: 0.83
embedding_fp16x8               mean time: 0.010128 ms, speedup: 0.91
embedding_fp16x8_packed        mean time: 0.009216 ms, speedup: 1.00
####################################################################################################
emd_size: 102400, emd_dim: 512, seq_len: 1024
torch                          mean time: 0.016208 ms
embedding                      mean time: 0.027488 ms, speedup: 0.59
embedding_fp32x4               mean time: 0.019584 ms, speedup: 0.83
embedding_fp32x4_packed        mean time: 0.015808 ms, speedup: 1.03
torch                          mean time: 0.011536 ms
embedding_half                 mean time: 0.023008 ms, speedup: 0.50
embedding_fp16x8               mean time: 0.019040 ms, speedup: 0.61
embedding_fp16x8_packed        mean time: 0.011680 ms, speedup: 0.99
```
