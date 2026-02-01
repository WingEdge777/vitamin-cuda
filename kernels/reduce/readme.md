# reduce_sum

## 说明

reduce_sum kernel

- [x] warp shuffle add
- [x] reduce_sum fp32/fp16 版
- [x] reduce_sum_fp16x2(fp16向量化)
- [x] reduce_sum_fp16x8_packed(fp16向量化, packed r/w)
- [x] reduce_sum int8 版
- [x] reduce_sum_i8x16_packed (int8向量化，packed r/w)
- [x] reduce_sum_i8x16_packed (int8向量化，packed r/w, dp4a, 相比torch朴素实现快几十倍)

## 测试

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### 输出

```bash
####################################################################################################
n: 1024, m: 1024
torch                          mean time: 0.021075 ms
reduce_sum                     mean time: 0.020603 ms
reduce_sum_fp32x4              mean time: 0.038919 ms
torch                          mean time: 0.021197 ms
reduce_sum_half                mean time: 0.035534 ms
reduce_sum_fp16x2              mean time: 0.035141 ms
reduce_sum_fp16x8_packed       mean time: 0.036065 ms
torch                          mean time: 0.034860 ms
reduce_sum_i8                  mean time: 0.021917 ms
reduce_sum_i8x16_packed        mean time: 0.019988 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.020126 ms
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.015169 ms
reduce_sum                     mean time: 0.029697 ms
reduce_sum_fp32x4              mean time: 0.018741 ms
torch                          mean time: 0.022959 ms
reduce_sum_half                mean time: 0.039188 ms
reduce_sum_fp16x2              mean time: 0.032351 ms
reduce_sum_fp16x8_packed       mean time: 0.034760 ms
torch                          mean time: 0.060970 ms
reduce_sum_i8                  mean time: 0.033742 ms
reduce_sum_i8x16_packed        mean time: 0.021347 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.022471 ms
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.026322 ms
reduce_sum                     mean time: 0.025661 ms
reduce_sum_fp32x4              mean time: 0.032358 ms
torch                          mean time: 0.020823 ms
reduce_sum_half                mean time: 0.036233 ms
reduce_sum_fp16x2              mean time: 0.034482 ms
reduce_sum_fp16x8_packed       mean time: 0.034508 ms
torch                          mean time: 0.165584 ms
reduce_sum_i8                  mean time: 0.029407 ms
reduce_sum_i8x16_packed        mean time: 0.025661 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.030774 ms
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.017779 ms
reduce_sum                     mean time: 0.027881 ms
reduce_sum_fp32x4              mean time: 0.032311 ms
torch                          mean time: 0.037946 ms
reduce_sum_half                mean time: 0.034173 ms
reduce_sum_fp16x2              mean time: 0.038173 ms
reduce_sum_fp16x8_packed       mean time: 0.027685 ms
torch                          mean time: 0.051577 ms
reduce_sum_i8                  mean time: 0.028256 ms
reduce_sum_i8x16_packed        mean time: 0.021990 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.019072 ms
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.034692 ms
reduce_sum                     mean time: 0.036503 ms
reduce_sum_fp32x4              mean time: 0.026699 ms
torch                          mean time: 0.019417 ms
reduce_sum_half                mean time: 0.031908 ms
reduce_sum_fp16x2              mean time: 0.046235 ms
reduce_sum_fp16x8_packed       mean time: 0.029801 ms
torch                          mean time: 0.178335 ms
reduce_sum_i8                  mean time: 0.033839 ms
reduce_sum_i8x16_packed        mean time: 0.022130 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.026875 ms
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.032749 ms
reduce_sum                     mean time: 0.046370 ms
reduce_sum_fp32x4              mean time: 0.029968 ms
torch                          mean time: 0.033251 ms
reduce_sum_half                mean time: 0.050191 ms
reduce_sum_fp16x2              mean time: 0.052436 ms
reduce_sum_fp16x8_packed       mean time: 0.039723 ms
torch                          mean time: 0.387652 ms
reduce_sum_i8                  mean time: 0.045651 ms
reduce_sum_i8x16_packed        mean time: 0.021912 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.031704 ms
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.028208 ms
reduce_sum                     mean time: 0.060309 ms
reduce_sum_fp32x4              mean time: 0.030974 ms
torch                          mean time: 0.018131 ms
reduce_sum_half                mean time: 0.040373 ms
reduce_sum_fp16x2              mean time: 0.032335 ms
reduce_sum_fp16x8_packed       mean time: 0.052160 ms
torch                          mean time: 0.161585 ms
reduce_sum_i8                  mean time: 0.032613 ms
reduce_sum_i8x16_packed        mean time: 0.021763 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.029059 ms
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.034557 ms
reduce_sum                     mean time: 0.046248 ms
reduce_sum_fp32x4              mean time: 0.027655 ms
torch                          mean time: 0.028626 ms
reduce_sum_half                mean time: 0.055176 ms
reduce_sum_fp16x2              mean time: 0.046623 ms
reduce_sum_fp16x8_packed       mean time: 0.045851 ms
torch                          mean time: 0.398549 ms
reduce_sum_i8                  mean time: 0.049189 ms
reduce_sum_i8x16_packed        mean time: 0.025204 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.022757 ms
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.200701 ms
reduce_sum                     mean time: 0.229292 ms
reduce_sum_fp32x4              mean time: 0.204083 ms
torch                          mean time: 0.038576 ms
reduce_sum_half                mean time: 0.096046 ms
reduce_sum_fp16x2              mean time: 0.077759 ms
reduce_sum_fp16x8_packed       mean time: 0.038822 ms
torch                          mean time: 0.894186 ms
reduce_sum_i8                  mean time: 0.086605 ms
reduce_sum_i8x16_packed        mean time: 0.037334 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.031133 ms
```
