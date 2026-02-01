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
torch                          mean time: 0.023403 ms
reduce_sum                     mean time: 0.028452 ms
reduce_sum_fp32x4              mean time: 0.029351 ms
torch                          mean time: 0.016217 ms
reduce_sum_half                mean time: 0.033940 ms
reduce_sum_fp16x2              mean time: 0.033102 ms
reduce_sum_fp16x8_packed       mean time: 0.033210 ms
torch                          mean time: 0.042865 ms
reduce_sum_i8                  mean time: 0.021183 ms
reduce_sum_i8x16_packed        mean time: 0.023986 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.019531 ms
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.015416 ms
reduce_sum                     mean time: 0.038015 ms
reduce_sum_fp32x4              mean time: 0.018968 ms
torch                          mean time: 0.031288 ms
reduce_sum_half                mean time: 0.070806 ms
reduce_sum_fp16x2              mean time: 0.030915 ms
reduce_sum_fp16x8_packed       mean time: 0.030066 ms
torch                          mean time: 0.057992 ms
reduce_sum_i8                  mean time: 0.025513 ms
reduce_sum_i8x16_packed        mean time: 0.023164 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.019998 ms
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.027649 ms
reduce_sum                     mean time: 0.027907 ms
reduce_sum_fp32x4              mean time: 0.028838 ms
torch                          mean time: 0.018981 ms
reduce_sum_half                mean time: 0.031663 ms
reduce_sum_fp16x2              mean time: 0.041312 ms
reduce_sum_fp16x8_packed       mean time: 0.038798 ms
torch                          mean time: 0.171779 ms
reduce_sum_i8                  mean time: 0.031569 ms
reduce_sum_i8x16_packed        mean time: 0.023740 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.030669 ms
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.018229 ms
reduce_sum                     mean time: 0.026867 ms
reduce_sum_fp32x4              mean time: 0.024228 ms
torch                          mean time: 0.025939 ms
reduce_sum_half                mean time: 0.033682 ms
reduce_sum_fp16x2              mean time: 0.043087 ms
reduce_sum_fp16x8_packed       mean time: 0.027980 ms
torch                          mean time: 0.058619 ms
reduce_sum_i8                  mean time: 0.025746 ms
reduce_sum_i8x16_packed        mean time: 0.018219 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.020443 ms
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.026930 ms
reduce_sum                     mean time: 0.033975 ms
reduce_sum_fp32x4              mean time: 0.028095 ms
torch                          mean time: 0.022600 ms
reduce_sum_half                mean time: 0.043450 ms
reduce_sum_fp16x2              mean time: 0.038425 ms
reduce_sum_fp16x8_packed       mean time: 0.043786 ms
torch                          mean time: 0.176716 ms
reduce_sum_i8                  mean time: 0.029667 ms
reduce_sum_i8x16_packed        mean time: 0.019931 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.022991 ms
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.032246 ms
reduce_sum                     mean time: 0.046270 ms
reduce_sum_fp32x4              mean time: 0.048891 ms
torch                          mean time: 0.030666 ms
reduce_sum_half                mean time: 0.049782 ms
reduce_sum_fp16x2              mean time: 0.035414 ms
reduce_sum_fp16x8_packed       mean time: 0.040630 ms
torch                          mean time: 0.429090 ms
reduce_sum_i8                  mean time: 0.047006 ms
reduce_sum_i8x16_packed        mean time: 0.020543 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.027605 ms
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.036654 ms
reduce_sum                     mean time: 0.033585 ms
reduce_sum_fp32x4              mean time: 0.028211 ms
torch                          mean time: 0.018402 ms
reduce_sum_half                mean time: 0.044369 ms
reduce_sum_fp16x2              mean time: 0.047849 ms
reduce_sum_fp16x8_packed       mean time: 0.033351 ms
torch                          mean time: 0.172470 ms
reduce_sum_i8                  mean time: 0.034088 ms
reduce_sum_i8x16_packed        mean time: 0.029995 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.021280 ms
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.057186 ms
reduce_sum                     mean time: 0.044477 ms
reduce_sum_fp32x4              mean time: 0.035164 ms
torch                          mean time: 0.033801 ms
reduce_sum_half                mean time: 0.053002 ms
reduce_sum_fp16x2              mean time: 0.056582 ms
reduce_sum_fp16x8_packed       mean time: 0.043832 ms
torch                          mean time: 0.434728 ms
reduce_sum_i8                  mean time: 0.055992 ms
reduce_sum_i8x16_packed        mean time: 0.025336 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.022764 ms
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.199757 ms
reduce_sum                     mean time: 0.223420 ms
reduce_sum_fp32x4              mean time: 0.201536 ms
torch                          mean time: 0.048468 ms
reduce_sum_half                mean time: 0.087097 ms
reduce_sum_fp16x2              mean time: 0.053115 ms
reduce_sum_fp16x8_packed       mean time: 0.043478 ms
torch                          mean time: 0.929718 ms
reduce_sum_i8                  mean time: 0.096258 ms
reduce_sum_i8x16_packed        mean time: 0.036727 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.030031 ms
```
