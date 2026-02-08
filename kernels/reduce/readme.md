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
- [x] reduce_sum_i8x64_packed (int8向量化，packed r/w, dp4a)

## 测试

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### 输出

```bash
n: 1024, m: 1024
torch                          mean time: 0.030733 ms
reduce_sum                     mean time: 0.030952 ms
reduce_sum_fp32x4              mean time: 0.031683 ms
torch                          mean time: 0.020447 ms
reduce_sum_half                mean time: 0.042702 ms
reduce_sum_fp16x2              mean time: 0.043842 ms
reduce_sum_fp16x8_packed       mean time: 0.042303 ms
torch                          mean time: 0.083829 ms
reduce_sum_i8                  mean time: 0.031489 ms
reduce_sum_i8x16_packed        mean time: 0.026856 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.025051 ms
reduce_sum_i8x64_packed_dp4a   mean time: 0.031824 ms
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.040677 ms
reduce_sum                     mean time: 0.034428 ms
reduce_sum_fp32x4              mean time: 0.031505 ms
torch                          mean time: 0.040303 ms
reduce_sum_half                mean time: 0.042109 ms
reduce_sum_fp16x2              mean time: 0.041581 ms
reduce_sum_fp16x8_packed       mean time: 0.038491 ms
torch                          mean time: 0.105637 ms
reduce_sum_i8                  mean time: 0.044122 ms
reduce_sum_i8x16_packed        mean time: 0.025087 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.029272 ms
reduce_sum_i8x64_packed_dp4a   mean time: 0.034867 ms
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.038996 ms
reduce_sum                     mean time: 0.044940 ms
reduce_sum_fp32x4              mean time: 0.049079 ms
torch                          mean time: 0.029841 ms
reduce_sum_half                mean time: 0.049029 ms
reduce_sum_fp16x2              mean time: 0.038618 ms
reduce_sum_fp16x8_packed       mean time: 0.049658 ms
torch                          mean time: 0.488492 ms
reduce_sum_i8                  mean time: 0.068718 ms
reduce_sum_i8x16_packed        mean time: 0.037476 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.023413 ms
reduce_sum_i8x64_packed_dp4a   mean time: 0.057119 ms
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.035642 ms
reduce_sum                     mean time: 0.035652 ms
reduce_sum_fp32x4              mean time: 0.059217 ms
torch                          mean time: 0.042993 ms
reduce_sum_half                mean time: 0.048548 ms
reduce_sum_fp16x2              mean time: 0.054250 ms
reduce_sum_fp16x8_packed       mean time: 0.040615 ms
torch                          mean time: 0.114117 ms
reduce_sum_i8                  mean time: 0.039824 ms
reduce_sum_i8x16_packed        mean time: 0.031822 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.028980 ms
reduce_sum_i8x64_packed_dp4a   mean time: 0.025808 ms
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.030043 ms
reduce_sum                     mean time: 0.049755 ms
reduce_sum_fp32x4              mean time: 0.041448 ms
torch                          mean time: 0.034299 ms
reduce_sum_half                mean time: 0.065313 ms
reduce_sum_fp16x2              mean time: 0.085600 ms
reduce_sum_fp16x8_packed       mean time: 0.055220 ms
torch                          mean time: 0.462205 ms
reduce_sum_i8                  mean time: 0.067580 ms
reduce_sum_i8x16_packed        mean time: 0.033659 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.029025 ms
reduce_sum_i8x64_packed_dp4a   mean time: 0.026428 ms
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.055541 ms
reduce_sum                     mean time: 0.081687 ms
reduce_sum_fp32x4              mean time: 0.060037 ms
torch                          mean time: 0.054856 ms
reduce_sum_half                mean time: 0.091652 ms
reduce_sum_fp16x2              mean time: 0.053021 ms
reduce_sum_fp16x8_packed       mean time: 0.040393 ms
torch                          mean time: 1.505230 ms
reduce_sum_i8                  mean time: 0.103972 ms
reduce_sum_i8x16_packed        mean time: 0.028185 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.031069 ms
reduce_sum_i8x64_packed_dp4a   mean time: 0.037486 ms
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.028938 ms
reduce_sum                     mean time: 0.047230 ms
reduce_sum_fp32x4              mean time: 0.034709 ms
torch                          mean time: 0.031319 ms
reduce_sum_half                mean time: 0.057039 ms
reduce_sum_fp16x2              mean time: 0.049052 ms
reduce_sum_fp16x8_packed       mean time: 0.041982 ms
torch                          mean time: 0.550463 ms
reduce_sum_i8                  mean time: 0.073581 ms
reduce_sum_i8x16_packed        mean time: 0.028400 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.019505 ms
reduce_sum_i8x64_packed_dp4a   mean time: 0.023245 ms
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.055313 ms
reduce_sum                     mean time: 0.082854 ms
reduce_sum_fp32x4              mean time: 0.054541 ms
torch                          mean time: 0.044961 ms
reduce_sum_half                mean time: 0.090271 ms
reduce_sum_fp16x2              mean time: 0.052674 ms
reduce_sum_fp16x8_packed       mean time: 0.038483 ms
torch                          mean time: 1.545086 ms
reduce_sum_i8                  mean time: 0.107700 ms
reduce_sum_i8x16_packed        mean time: 0.036750 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.029261 ms
reduce_sum_i8x64_packed_dp4a   mean time: 0.024693 ms
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.548919 ms
reduce_sum                     mean time: 0.892459 ms
reduce_sum_fp32x4              mean time: 0.627701 ms
torch                          mean time: 0.124138 ms
reduce_sum_half                mean time: 0.149437 ms
reduce_sum_fp16x2              mean time: 0.086820 ms
reduce_sum_fp16x8_packed       mean time: 0.079791 ms
torch                          mean time: 3.517789 ms
reduce_sum_i8                  mean time: 0.167590 ms
reduce_sum_i8x16_packed        mean time: 0.040536 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.046063 ms
reduce_sum_i8x64_packed_dp4a   mean time: 0.042408 ms
```
