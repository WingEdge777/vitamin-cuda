# reduce_sum

## Overview

reduce-sum kernels.

- [x] warp shuffle reduction
- [x] reduce_sum — FP32 / FP16
- [x] reduce_sum_fp16x2 — vectorized FP16
- [x] reduce_sum_fp16x8_packed — vectorized FP16, packed r/w
- [x] reduce_sum — INT8
- [x] reduce_sum_i8x16_packed — vectorized INT8, packed r/w
- [x] reduce_sum_i8x16_packed_dp4a — INT8, packed r/w, `dp4a` (orders of magnitude vs naive Torch)
- [x] reduce_sum_i8x64_packed_dp4a — INT8, packed r/w, `dp4a`

## Run tests

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### Sample output

```bash
####################################################################################################
n: 1024, m: 1024
torch                          mean time: 0.030208 ms
reduce_sum                     mean time: 0.050192 ms, speedup: 0.60
reduce_sum_fp32x4              mean time: 0.024016 ms, speedup: 1.26
torch                          mean time: 0.017328 ms
reduce_sum_half                mean time: 0.067216 ms, speedup: 0.26
reduce_sum_fp16x2              mean time: 0.045488 ms, speedup: 0.38
reduce_sum_fp16x8_packed       mean time: 0.042624 ms, speedup: 0.41
torch                          mean time: 0.142432 ms
reduce_sum_i8                  mean time: 0.068816 ms, speedup: 2.07
reduce_sum_i8x16_packed        mean time: 0.040192 ms, speedup: 3.54
reduce_sum_i8x16_packed_dp4a   mean time: 0.038592 ms, speedup: 3.69
reduce_sum_i8x64_packed_dp4a   mean time: 0.038672 ms, speedup: 3.68
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.115504 ms
reduce_sum                     mean time: 0.082336 ms, speedup: 1.40
reduce_sum_fp32x4              mean time: 0.054816 ms, speedup: 2.11
torch                          mean time: 0.035056 ms
reduce_sum_half                mean time: 0.083424 ms, speedup: 0.42
reduce_sum_fp16x2              mean time: 0.052576 ms, speedup: 0.67
reduce_sum_fp16x8_packed       mean time: 0.040432 ms, speedup: 0.87
torch                          mean time: 0.191728 ms
reduce_sum_i8                  mean time: 0.078064 ms, speedup: 2.46
reduce_sum_i8x16_packed        mean time: 0.027616 ms, speedup: 6.94
reduce_sum_i8x16_packed_dp4a   mean time: 0.032768 ms, speedup: 5.85
reduce_sum_i8x64_packed_dp4a   mean time: 0.045040 ms, speedup: 4.26
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.183616 ms
reduce_sum                     mean time: 0.244608 ms, speedup: 0.75
reduce_sum_fp32x4              mean time: 0.171328 ms, speedup: 1.07
torch                          mean time: 0.079296 ms
reduce_sum_half                mean time: 0.153792 ms, speedup: 0.52
reduce_sum_fp16x2              mean time: 0.100000 ms, speedup: 0.79
reduce_sum_fp16x8_packed       mean time: 0.070576 ms, speedup: 1.12
torch                          mean time: 0.733424 ms
reduce_sum_i8                  mean time: 0.146000 ms, speedup: 5.02
reduce_sum_i8x16_packed        mean time: 0.042784 ms, speedup: 17.14
reduce_sum_i8x16_packed_dp4a   mean time: 0.044016 ms, speedup: 16.66
reduce_sum_i8x64_packed_dp4a   mean time: 0.036656 ms, speedup: 20.01
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.051488 ms
reduce_sum                     mean time: 0.090000 ms, speedup: 0.57
reduce_sum_fp32x4              mean time: 0.064192 ms, speedup: 0.80
torch                          mean time: 0.045008 ms
reduce_sum_half                mean time: 0.103456 ms, speedup: 0.44
reduce_sum_fp16x2              mean time: 0.106064 ms, speedup: 0.42
reduce_sum_fp16x8_packed       mean time: 0.082912 ms, speedup: 0.54
torch                          mean time: 0.538352 ms
reduce_sum_i8                  mean time: 0.151168 ms, speedup: 3.56
reduce_sum_i8x16_packed        mean time: 0.042368 ms, speedup: 12.71
reduce_sum_i8x16_packed_dp4a   mean time: 0.034464 ms, speedup: 15.62
reduce_sum_i8x64_packed_dp4a   mean time: 0.028736 ms, speedup: 18.73
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.082800 ms
reduce_sum                     mean time: 0.144240 ms, speedup: 0.57
reduce_sum_fp32x4              mean time: 0.094384 ms, speedup: 0.88
torch                          mean time: 0.072288 ms
reduce_sum_half                mean time: 0.150384 ms, speedup: 0.48
reduce_sum_fp16x2              mean time: 0.103584 ms, speedup: 0.70
reduce_sum_fp16x8_packed       mean time: 0.067072 ms, speedup: 1.08
torch                          mean time: 0.606768 ms
reduce_sum_i8                  mean time: 0.139008 ms, speedup: 4.36
reduce_sum_i8x16_packed        mean time: 0.049360 ms, speedup: 12.29
reduce_sum_i8x16_packed_dp4a   mean time: 0.050000 ms, speedup: 12.14
reduce_sum_i8x64_packed_dp4a   mean time: 0.039504 ms, speedup: 15.36
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.144448 ms
reduce_sum                     mean time: 0.446992 ms, speedup: 0.32
reduce_sum_fp32x4              mean time: 0.303264 ms, speedup: 0.48
torch                          mean time: 0.183744 ms
reduce_sum_half                mean time: 0.444448 ms, speedup: 0.41
reduce_sum_fp16x2              mean time: 0.260560 ms, speedup: 0.71
reduce_sum_fp16x8_packed       mean time: 0.180608 ms, speedup: 1.02
torch                          mean time: 1.414496 ms
reduce_sum_i8                  mean time: 0.231472 ms, speedup: 6.11
reduce_sum_i8x16_packed        mean time: 0.062160 ms, speedup: 22.76
reduce_sum_i8x16_packed_dp4a   mean time: 0.069216 ms, speedup: 20.44
reduce_sum_i8x64_packed_dp4a   mean time: 0.053280 ms, speedup: 26.55
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.082880 ms
reduce_sum                     mean time: 0.181920 ms, speedup: 0.46
reduce_sum_fp32x4              mean time: 0.157328 ms, speedup: 0.53
torch                          mean time: 0.091312 ms
reduce_sum_half                mean time: 0.161456 ms, speedup: 0.57
reduce_sum_fp16x2              mean time: 0.097456 ms, speedup: 0.94
reduce_sum_fp16x8_packed       mean time: 0.071376 ms, speedup: 1.28
torch                          mean time: 0.549680 ms
reduce_sum_i8                  mean time: 0.132848 ms, speedup: 4.14
reduce_sum_i8x16_packed        mean time: 0.044288 ms, speedup: 12.41
reduce_sum_i8x16_packed_dp4a   mean time: 0.048960 ms, speedup: 11.23
reduce_sum_i8x64_packed_dp4a   mean time: 0.036144 ms, speedup: 15.21
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.307856 ms
reduce_sum                     mean time: 0.419456 ms, speedup: 0.73
reduce_sum_fp32x4              mean time: 0.204736 ms, speedup: 1.50
torch                          mean time: 0.128000 ms
reduce_sum_half                mean time: 0.271152 ms, speedup: 0.47
reduce_sum_fp16x2              mean time: 0.159232 ms, speedup: 0.80
reduce_sum_fp16x8_packed       mean time: 0.098464 ms, speedup: 1.30
torch                          mean time: 1.302592 ms
reduce_sum_i8                  mean time: 0.265984 ms, speedup: 4.90
reduce_sum_i8x16_packed        mean time: 0.077712 ms, speedup: 16.76
reduce_sum_i8x16_packed_dp4a   mean time: 0.072288 ms, speedup: 18.02
reduce_sum_i8x64_packed_dp4a   mean time: 0.057296 ms, speedup: 22.73
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.411328 ms
reduce_sum                     mean time: 0.635216 ms, speedup: 0.65
reduce_sum_fp32x4              mean time: 0.430384 ms, speedup: 0.96
torch                          mean time: 0.255440 ms
reduce_sum_half                mean time: 0.547088 ms, speedup: 0.47
reduce_sum_fp16x2              mean time: 0.332880 ms, speedup: 0.77
reduce_sum_fp16x8_packed       mean time: 0.179840 ms, speedup: 1.42
torch                          mean time: 3.457664 ms
reduce_sum_i8                  mean time: 0.488592 ms, speedup: 7.08
reduce_sum_i8x16_packed        mean time: 0.114880 ms, speedup: 30.10
reduce_sum_i8x16_packed_dp4a   mean time: 0.127216 ms, speedup: 27.18
reduce_sum_i8x64_packed_dp4a   mean time: 0.104192 ms, speedup: 33.19
```
