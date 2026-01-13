# Elementwise

## 说明

sigmoid kernel

- [x] sigmoid fp32/fp16 版
- [x] sigmoid_fp16x2(fp16向量化)
- [x] sigmoid_fp16x8(fp16向量化)
- [x] sigmoid_fp16x8(fp16向量化, packeds r/add/w)
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
torch                          mean time: 0.012303 ms
sigmoid                        mean time: 0.013582 ms
sigmoid_fp32x4                 mean time: 0.009824 ms
torch                          mean time: 0.010448 ms
sigmoid_half                   mean time: 0.013474 ms
sigmoid_fp16x2                 mean time: 0.012172 ms
sigmoid_fp16x8                 mean time: 0.016625 ms
sigmoid_fp16x8_packed          mean time: 0.010795 ms
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.013498 ms
sigmoid                        mean time: 0.024879 ms
sigmoid_fp32x4                 mean time: 0.017798 ms
torch                          mean time: 0.011517 ms
sigmoid_half                   mean time: 0.028724 ms
sigmoid_fp16x2                 mean time: 0.020642 ms
sigmoid_fp16x8                 mean time: 0.019249 ms
sigmoid_fp16x8_packed          mean time: 0.015884 ms
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.025210 ms
sigmoid                        mean time: 0.046091 ms
sigmoid_fp32x4                 mean time: 0.030207 ms
torch                          mean time: 0.024422 ms
sigmoid_half                   mean time: 0.049791 ms
sigmoid_fp16x2                 mean time: 0.036503 ms
sigmoid_fp16x8                 mean time: 0.035755 ms
sigmoid_fp16x8_packed          mean time: 0.025184 ms
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.017261 ms
sigmoid                        mean time: 0.020954 ms
sigmoid_fp32x4                 mean time: 0.017286 ms
torch                          mean time: 0.011326 ms
sigmoid_half                   mean time: 0.027373 ms
sigmoid_fp16x2                 mean time: 0.016379 ms
sigmoid_fp16x8                 mean time: 0.024208 ms
sigmoid_fp16x8_packed          mean time: 0.011158 ms
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.024316 ms
sigmoid                        mean time: 0.051483 ms
sigmoid_fp32x4                 mean time: 0.030438 ms
torch                          mean time: 0.026023 ms
sigmoid_half                   mean time: 0.050677 ms
sigmoid_fp16x2                 mean time: 0.031362 ms
sigmoid_fp16x8                 mean time: 0.041672 ms
sigmoid_fp16x8_packed          mean time: 0.026203 ms
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.205353 ms
sigmoid                        mean time: 0.215819 ms
sigmoid_fp32x4                 mean time: 0.201072 ms
torch                          mean time: 0.038648 ms
sigmoid_half                   mean time: 0.136292 ms
sigmoid_fp16x2                 mean time: 0.079532 ms
sigmoid_fp16x8                 mean time: 0.103854 ms
sigmoid_fp16x8_packed          mean time: 0.046987 ms
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.038961 ms
sigmoid                        mean time: 0.061337 ms
sigmoid_fp32x4                 mean time: 0.031756 ms
torch                          mean time: 0.026403 ms
sigmoid_half                   mean time: 0.061747 ms
sigmoid_fp16x2                 mean time: 0.034573 ms
sigmoid_fp16x8                 mean time: 0.041457 ms
sigmoid_fp16x8_packed          mean time: 0.025997 ms
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.208895 ms
sigmoid                        mean time: 0.217116 ms
sigmoid_fp32x4                 mean time: 0.201629 ms
torch                          mean time: 0.033038 ms
sigmoid_half                   mean time: 0.116040 ms
sigmoid_fp16x2                 mean time: 0.069979 ms
sigmoid_fp16x8                 mean time: 0.084627 ms
sigmoid_fp16x8_packed          mean time: 0.054925 ms
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.402722 ms
sigmoid                        mean time: 0.418494 ms
sigmoid_fp32x4                 mean time: 0.409435 ms
torch                          mean time: 0.210508 ms
sigmoid_half                   mean time: 0.330328 ms
sigmoid_fp16x2                 mean time: 0.229985 ms
sigmoid_fp16x8                 mean time: 0.206905 ms
sigmoid_fp16x8_packed          mean time: 0.209094 ms
```
