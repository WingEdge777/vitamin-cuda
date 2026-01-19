# silu

## 说明

silu kernel

- [x] silu fp32/fp16 版
- [x] silu_fp16x2(fp16向量化)
- [x] silu_fp16x8(fp16向量化)
- [x] silu_fp16x8(fp16向量化, packed r/w)
- [x] pytorch op bindings && diff check

## 测试

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### 输出

```bash
####################################################################################################
n: 1024, m: 1024
torch                          mean time: 0.012343 ms
silu                           mean time: 0.012918 ms
silu_fp32x4                    mean time: 0.014699 ms
torch                          mean time: 0.017207 ms
silu_half                      mean time: 0.014085 ms
silu_fp16x2                    mean time: 0.018678 ms
silu_fp16x8                    mean time: 0.017202 ms
silu_fp16x8_packed             mean time: 0.015122 ms
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.020994 ms
silu                           mean time: 0.029243 ms
silu_fp32x4                    mean time: 0.019567 ms
torch                          mean time: 0.026718 ms
silu_half                      mean time: 0.028427 ms
silu_fp16x2                    mean time: 0.015557 ms
silu_fp16x8                    mean time: 0.023715 ms
silu_fp16x8_packed             mean time: 0.010728 ms
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.039425 ms
silu                           mean time: 0.062426 ms
silu_fp32x4                    mean time: 0.034703 ms
torch                          mean time: 0.049607 ms
silu_half                      mean time: 0.052124 ms
silu_fp16x2                    mean time: 0.031922 ms
silu_fp16x8                    mean time: 0.036576 ms
silu_fp16x8_packed             mean time: 0.018207 ms
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.023970 ms
silu                           mean time: 0.028181 ms
silu_fp32x4                    mean time: 0.019510 ms
torch                          mean time: 0.022307 ms
silu_half                      mean time: 0.029076 ms
silu_fp16x2                    mean time: 0.014349 ms
silu_fp16x8                    mean time: 0.028963 ms
silu_fp16x8_packed             mean time: 0.011702 ms
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.039579 ms
silu                           mean time: 0.062266 ms
silu_fp32x4                    mean time: 0.035576 ms
torch                          mean time: 0.048203 ms
silu_half                      mean time: 0.059454 ms
silu_fp16x2                    mean time: 0.030193 ms
silu_fp16x8                    mean time: 0.038039 ms
silu_fp16x8_packed             mean time: 0.018093 ms
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.104354 ms
silu                           mean time: 0.222844 ms
silu_fp32x4                    mean time: 0.211169 ms
torch                          mean time: 0.100642 ms
silu_half                      mean time: 0.121104 ms
silu_fp16x2                    mean time: 0.053788 ms
silu_fp16x8                    mean time: 0.093420 ms
silu_fp16x8_packed             mean time: 0.031881 ms
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.047348 ms
silu                           mean time: 0.063662 ms
silu_fp32x4                    mean time: 0.042660 ms
torch                          mean time: 0.052953 ms
silu_half                      mean time: 0.061517 ms
silu_fp16x2                    mean time: 0.030703 ms
silu_fp16x8                    mean time: 0.037476 ms
silu_fp16x8_packed             mean time: 0.027437 ms
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.110683 ms
silu                           mean time: 0.226239 ms
silu_fp32x4                    mean time: 0.210940 ms
torch                          mean time: 0.088213 ms
silu_half                      mean time: 0.109631 ms
silu_fp16x2                    mean time: 0.063520 ms
silu_fp16x8                    mean time: 0.087788 ms
silu_fp16x8_packed             mean time: 0.027862 ms
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.408961 ms
silu                           mean time: 0.443659 ms
silu_fp32x4                    mean time: 0.412657 ms
torch                          mean time: 0.192371 ms
silu_half                      mean time: 0.378837 ms
silu_fp16x2                    mean time: 0.234156 ms
silu_fp16x8                    mean time: 0.214144 ms
silu_fp16x8_packed             mean time: 0.211251 ms
```
