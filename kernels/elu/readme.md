# Elu

## 说明

elu kernel

- [x] elu fp32/fp16 版
- [x] elu_fp16x2(fp16向量化)
- [x] elu_fp16x8(fp16向量化)
- [x] elu_fp16x8(fp16向量化, packed r/w, half2 近两倍提升)
- [x] pytorch op bindings

## 测试

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### 输出

```bash
####################################################################################################
n: 1024, m: 1024
torch                          mean time: 0.016135 ms
elu                            mean time: 0.020419 ms
elu_fp32x4                     mean time: 0.009367 ms
torch                          mean time: 0.018720 ms
elu_half                       mean time: 0.023111 ms
elu_fp16x2                     mean time: 0.014962 ms
elu_fp16x8                     mean time: 0.018292 ms
elu_fp16x8_packed              mean time: 0.014132 ms
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.026670 ms
elu                            mean time: 0.031063 ms
elu_fp32x4                     mean time: 0.018426 ms
torch                          mean time: 0.027915 ms
elu_half                       mean time: 0.043214 ms
elu_fp16x2                     mean time: 0.025848 ms
elu_fp16x8                     mean time: 0.037226 ms
elu_fp16x8_packed              mean time: 0.018488 ms
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.033110 ms
elu                            mean time: 0.212075 ms
elu_fp32x4                     mean time: 0.196537 ms
torch                          mean time: 0.130637 ms
elu_half                       mean time: 0.090890 ms
elu_fp16x2                     mean time: 0.041907 ms
elu_fp16x8                     mean time: 0.067041 ms
elu_fp16x8_packed              mean time: 0.025889 ms
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.037810 ms
elu                            mean time: 0.060480 ms
elu_fp32x4                     mean time: 0.038936 ms
torch                          mean time: 0.060284 ms
elu_half                       mean time: 0.082076 ms
elu_fp16x2                     mean time: 0.038596 ms
elu_fp16x8                     mean time: 0.068374 ms
elu_fp16x8_packed              mean time: 0.023226 ms
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.187787 ms
elu                            mean time: 0.440192 ms
elu_fp32x4                     mean time: 0.265210 ms
torch                          mean time: 0.098931 ms
elu_half                       mean time: 0.091958 ms
elu_fp16x2                     mean time: 0.042308 ms
elu_fp16x8                     mean time: 0.069547 ms
elu_fp16x8_packed              mean time: 0.035841 ms
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.458813 ms
elu                            mean time: 0.929212 ms
elu_fp32x4                     mean time: 0.583942 ms
torch                          mean time: 0.597380 ms
elu_half                       mean time: 0.328474 ms
elu_fp16x2                     mean time: 0.207090 ms
elu_fp16x8                     mean time: 0.410421 ms
elu_fp16x8_packed              mean time: 0.260892 ms
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.295651 ms
elu                            mean time: 0.439934 ms
elu_fp32x4                     mean time: 0.290076 ms
torch                          mean time: 0.130208 ms
elu_half                       mean time: 0.085979 ms
elu_fp16x2                     mean time: 0.043027 ms
elu_fp16x8                     mean time: 0.067967 ms
elu_fp16x8_packed              mean time: 0.024264 ms
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.371306 ms
elu                            mean time: 0.870806 ms
elu_fp32x4                     mean time: 0.583772 ms
torch                          mean time: 0.474186 ms
elu_half                       mean time: 0.408232 ms
elu_fp16x2                     mean time: 0.388984 ms
elu_fp16x8                     mean time: 0.428567 ms
elu_fp16x8_packed              mean time: 0.249185 ms
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 1.321477 ms
elu                            mean time: 1.854496 ms
elu_fp32x4                     mean time: 1.142058 ms
torch                          mean time: 0.949690 ms
elu_half                       mean time: 0.959096 ms
elu_fp16x2                     mean time: 0.958181 ms
elu_fp16x8                     mean time: 1.103509 ms
elu_fp16x8_packed              mean time: 0.545815 ms
```
