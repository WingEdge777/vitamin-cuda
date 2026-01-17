# Relu

## 说明

relu kernel

- [x] relu fp32/fp16 版
- [x] relu_fp16x2(fp16向量化)
- [x] relu_fp16x8(fp16向量化)
- [x] relu_fp16x8(fp16向量化, packed r/w)
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
torch                          mean time: 0.019082 ms
relu                           mean time: 0.018298 ms
relu_fp32x4                    mean time: 0.012728 ms
torch                          mean time: 0.011281 ms
relu_half                      mean time: 0.016121 ms
relu_fp16x2                    mean time: 0.012518 ms
relu_fp16x8                    mean time: 0.014536 ms
relu_fp16x8_packed             mean time: 0.011192 ms
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.017387 ms
relu                           mean time: 0.029374 ms
relu_fp32x4                    mean time: 0.024635 ms
torch                          mean time: 0.021073 ms
relu_half                      mean time: 0.033957 ms
relu_fp16x2                    mean time: 0.018727 ms
relu_fp16x8                    mean time: 0.022552 ms
relu_fp16x8_packed             mean time: 0.016796 ms
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.030384 ms
relu                           mean time: 0.091705 ms
relu_fp32x4                    mean time: 0.207337 ms
torch                          mean time: 0.079463 ms
relu_half                      mean time: 0.092208 ms
relu_fp16x2                    mean time: 0.036990 ms
relu_fp16x8                    mean time: 0.042954 ms
relu_fp16x8_packed             mean time: 0.025755 ms
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.022762 ms
relu                           mean time: 0.034036 ms
relu_fp32x4                    mean time: 0.028480 ms
torch                          mean time: 0.016038 ms
relu_half                      mean time: 0.037245 ms
relu_fp16x2                    mean time: 0.022274 ms
relu_fp16x8                    mean time: 0.025401 ms
relu_fp16x8_packed             mean time: 0.016016 ms
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.036854 ms
relu                           mean time: 0.290323 ms
relu_fp32x4                    mean time: 0.293166 ms
torch                          mean time: 0.112240 ms
relu_half                      mean time: 0.135247 ms
relu_fp16x2                    mean time: 0.049804 ms
relu_fp16x8                    mean time: 0.073521 ms
relu_fp16x8_packed             mean time: 0.035705 ms
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.480532 ms
relu                           mean time: 0.875122 ms
relu_fp32x4                    mean time: 0.572695 ms
torch                          mean time: 0.296135 ms
relu_half                      mean time: 0.492179 ms
relu_fp16x2                    mean time: 0.201990 ms
relu_fp16x8                    mean time: 0.272710 ms
relu_fp16x8_packed             mean time: 0.191736 ms
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.175733 ms
relu                           mean time: 0.295491 ms
relu_fp32x4                    mean time: 0.266913 ms
torch                          mean time: 0.089539 ms
relu_half                      mean time: 0.095054 ms
relu_fp16x2                    mean time: 0.038160 ms
relu_fp16x8                    mean time: 0.044555 ms
relu_fp16x8_packed             mean time: 0.024592 ms
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.507048 ms
relu                           mean time: 0.864296 ms
relu_fp32x4                    mean time: 0.571961 ms
torch                          mean time: 0.292946 ms
relu_half                      mean time: 0.314549 ms
relu_fp16x2                    mean time: 0.283704 ms
relu_fp16x8                    mean time: 0.391538 ms
relu_fp16x8_packed             mean time: 0.240592 ms
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 1.028275 ms
relu                           mean time: 1.750600 ms
relu_fp32x4                    mean time: 1.124701 ms
torch                          mean time: 0.587842 ms
relu_half                      mean time: 1.215562 ms
relu_fp16x2                    mean time: 0.676718 ms
relu_fp16x8                    mean time: 1.044594 ms
relu_fp16x8_packed             mean time: 0.553104 ms
```
