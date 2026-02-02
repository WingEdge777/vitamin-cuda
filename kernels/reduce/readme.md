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
####################################################################################################
n: 1024, m: 1024
torch                          mean time: 0.020551 ms
reduce_sum                     mean time: 0.018767 ms
reduce_sum_fp32x4              mean time: 0.021513 ms
torch                          mean time: 0.017108 ms
reduce_sum_half                mean time: 0.036333 ms
reduce_sum_fp16x2              mean time: 0.030213 ms
reduce_sum_fp16x8_packed       mean time: 0.044891 ms
torch                          mean time: 0.035277 ms
reduce_sum_i8                  mean time: 0.021461 ms
reduce_sum_i8x16_packed        mean time: 0.020919 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.027804 ms
reduce_sum_i8x64_packed_dp4a   mean time: 0.022531 ms
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.024939 ms
reduce_sum                     mean time: 0.032620 ms
reduce_sum_fp32x4              mean time: 0.022450 ms
torch                          mean time: 0.023552 ms
reduce_sum_half                mean time: 0.034781 ms
reduce_sum_fp16x2              mean time: 0.039238 ms
reduce_sum_fp16x8_packed       mean time: 0.032223 ms
torch                          mean time: 0.058835 ms
reduce_sum_i8                  mean time: 0.029611 ms
reduce_sum_i8x16_packed        mean time: 0.018860 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.021321 ms
reduce_sum_i8x64_packed_dp4a   mean time: 0.019896 ms
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.029149 ms
reduce_sum                     mean time: 0.034981 ms
reduce_sum_fp32x4              mean time: 0.032027 ms
torch                          mean time: 0.014932 ms
reduce_sum_half                mean time: 0.035516 ms
reduce_sum_fp16x2              mean time: 0.039639 ms
reduce_sum_fp16x8_packed       mean time: 0.031909 ms
torch                          mean time: 0.175100 ms
reduce_sum_i8                  mean time: 0.031851 ms
reduce_sum_i8x16_packed        mean time: 0.025426 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.024994 ms
reduce_sum_i8x64_packed_dp4a   mean time: 0.025783 ms
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.018805 ms
reduce_sum                     mean time: 0.027645 ms
reduce_sum_fp32x4              mean time: 0.021190 ms
torch                          mean time: 0.023247 ms
reduce_sum_half                mean time: 0.041256 ms
reduce_sum_fp16x2              mean time: 0.032527 ms
reduce_sum_fp16x8_packed       mean time: 0.030985 ms
torch                          mean time: 0.058087 ms
reduce_sum_i8                  mean time: 0.030724 ms
reduce_sum_i8x16_packed        mean time: 0.039050 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.020959 ms
reduce_sum_i8x64_packed_dp4a   mean time: 0.019963 ms
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.032431 ms
reduce_sum                     mean time: 0.030082 ms
reduce_sum_fp32x4              mean time: 0.031209 ms
torch                          mean time: 0.016963 ms
reduce_sum_half                mean time: 0.037880 ms
reduce_sum_fp16x2              mean time: 0.039626 ms
reduce_sum_fp16x8_packed       mean time: 0.037719 ms
torch                          mean time: 0.171252 ms
reduce_sum_i8                  mean time: 0.035753 ms
reduce_sum_i8x16_packed        mean time: 0.031656 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.027456 ms
reduce_sum_i8x64_packed_dp4a   mean time: 0.019943 ms
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.032501 ms
reduce_sum                     mean time: 0.050994 ms
reduce_sum_fp32x4              mean time: 0.052937 ms
torch                          mean time: 0.050790 ms
reduce_sum_half                mean time: 0.056073 ms
reduce_sum_fp16x2              mean time: 0.044301 ms
reduce_sum_fp16x8_packed       mean time: 0.039778 ms
torch                          mean time: 0.404670 ms
reduce_sum_i8                  mean time: 0.049248 ms
reduce_sum_i8x16_packed        mean time: 0.030979 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.021482 ms
reduce_sum_i8x64_packed_dp4a   mean time: 0.019908 ms
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.031419 ms
reduce_sum                     mean time: 0.032096 ms
reduce_sum_fp32x4              mean time: 0.034184 ms
torch                          mean time: 0.020221 ms
reduce_sum_half                mean time: 0.047724 ms
reduce_sum_fp16x2              mean time: 0.047936 ms
reduce_sum_fp16x8_packed       mean time: 0.028529 ms
torch                          mean time: 0.175490 ms
reduce_sum_i8                  mean time: 0.043211 ms
reduce_sum_i8x16_packed        mean time: 0.022674 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.027911 ms
reduce_sum_i8x64_packed_dp4a   mean time: 0.025020 ms
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.043897 ms
reduce_sum                     mean time: 0.054749 ms
reduce_sum_fp32x4              mean time: 0.033586 ms
torch                          mean time: 0.031091 ms
reduce_sum_half                mean time: 0.052396 ms
reduce_sum_fp16x2              mean time: 0.041730 ms
reduce_sum_fp16x8_packed       mean time: 0.044537 ms
torch                          mean time: 0.410589 ms
reduce_sum_i8                  mean time: 0.046099 ms
reduce_sum_i8x16_packed        mean time: 0.021789 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.023361 ms
reduce_sum_i8x64_packed_dp4a   mean time: 0.030942 ms
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.237330 ms
reduce_sum                     mean time: 0.240531 ms
reduce_sum_fp32x4              mean time: 0.210875 ms
torch                          mean time: 0.042151 ms
reduce_sum_half                mean time: 0.093258 ms
reduce_sum_fp16x2              mean time: 0.072626 ms
reduce_sum_fp16x8_packed       mean time: 0.045490 ms
torch                          mean time: 0.923436 ms
reduce_sum_i8                  mean time: 0.090087 ms
reduce_sum_i8x16_packed        mean time: 0.038203 ms
reduce_sum_i8x16_packed_dp4a   mean time: 0.033357 ms
reduce_sum_i8x64_packed_dp4a   mean time: 0.024007 ms
```
