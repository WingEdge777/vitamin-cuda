# Elementwise

## 说明

涵盖了elementwise的加法kernel,

- [x] elementwise_add fp32/fp16 版
- [x] elementwise_add_fp16x2(fp16向量化)
- [x] elementwise_add_fp16x8(fp16向量化)
- [x] elementwise_add_fp16x8(fp16向量化, packeds r/add/w)
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
torch                          mean time: 0.012049 ms
elementwise_add                mean time: 0.013310 ms
elementwise_add_fp32x4         mean time: 0.028203 ms
torch                          mean time: 0.013195 ms
ele_add_half                   mean time: 0.013967 ms
elementwise_add_fp16x2         mean time: 0.011470 ms
elementwise_add_fp16x8         mean time: 0.011240 ms
elementwise_add_fp16x8_packed  mean time: 0.009198 ms
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.015162 ms
elementwise_add                mean time: 0.024646 ms
elementwise_add_fp32x4         mean time: 0.015189 ms
torch                          mean time: 0.011911 ms
ele_add_half                   mean time: 0.021978 ms
elementwise_add_fp16x2         mean time: 0.013251 ms
elementwise_add_fp16x8         mean time: 0.015358 ms
elementwise_add_fp16x8_packed  mean time: 0.011225 ms
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.120534 ms
elementwise_add                mean time: 0.121183 ms
elementwise_add_fp32x4         mean time: 0.119958 ms
torch                          mean time: 0.015178 ms
ele_add_half                   mean time: 0.042252 ms
elementwise_add_fp16x2         mean time: 0.023469 ms
elementwise_add_fp16x8         mean time: 0.025564 ms
elementwise_add_fp16x8_packed  mean time: 0.015207 ms
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.015733 ms
elementwise_add                mean time: 0.023226 ms
elementwise_add_fp32x4         mean time: 0.015195 ms
torch                          mean time: 0.012475 ms
ele_add_half                   mean time: 0.022930 ms
elementwise_add_fp16x2         mean time: 0.013531 ms
elementwise_add_fp16x8         mean time: 0.014809 ms
elementwise_add_fp16x8_packed  mean time: 0.012180 ms
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.120688 ms
elementwise_add                mean time: 0.122921 ms
elementwise_add_fp32x4         mean time: 0.120237 ms
torch                          mean time: 0.017869 ms
ele_add_half                   mean time: 0.042757 ms
elementwise_add_fp16x2         mean time: 0.023475 ms
elementwise_add_fp16x8         mean time: 0.025447 ms
elementwise_add_fp16x8_packed  mean time: 0.016079 ms
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.300478 ms
elementwise_add                mean time: 0.303731 ms
elementwise_add_fp32x4         mean time: 0.298697 ms
torch                          mean time: 0.119616 ms
ele_add_half                   mean time: 0.179682 ms
elementwise_add_fp16x2         mean time: 0.130236 ms
elementwise_add_fp16x8         mean time: 0.126467 ms
elementwise_add_fp16x8_packed  mean time: 0.119159 ms
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.120143 ms
elementwise_add                mean time: 0.128705 ms
elementwise_add_fp32x4         mean time: 0.127114 ms
torch                          mean time: 0.017406 ms
ele_add_half                   mean time: 0.046127 ms
elementwise_add_fp16x2         mean time: 0.024073 ms
elementwise_add_fp16x8         mean time: 0.025630 ms
elementwise_add_fp16x8_packed  mean time: 0.015484 ms
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.319201 ms
elementwise_add                mean time: 0.300200 ms
elementwise_add_fp32x4         mean time: 0.297716 ms
torch                          mean time: 0.119841 ms
ele_add_half                   mean time: 0.177679 ms
elementwise_add_fp16x2         mean time: 0.121720 ms
elementwise_add_fp16x8         mean time: 0.119427 ms
elementwise_add_fp16x8_packed  mean time: 0.119342 ms
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.595576 ms
elementwise_add                mean time: 0.649621 ms
elementwise_add_fp32x4         mean time: 0.603019 ms
torch                          mean time: 0.323687 ms
ele_add_half                   mean time: 0.369825 ms
elementwise_add_fp16x2         mean time: 0.298279 ms
elementwise_add_fp16x8         mean time: 0.312507 ms
elementwise_add_fp16x8_packed  mean time: 0.295677 ms
```
