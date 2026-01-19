# Sigmoid

## 说明

sigmoid kernel

- [x] sigmoid fp32/fp16 版
- [x] sigmoid_fp16x2(fp16向量化)
- [x] sigmoid_fp16x8(fp16向量化)
- [x] sigmoid_fp16x8(fp16向量化, packed r/w)
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
torch                          mean time: 0.011463 ms
sigmoid                        mean time: 0.012771 ms
sigmoid_fp32x4                 mean time: 0.027378 ms
torch                          mean time: 0.009884 ms
sigmoid_half                   mean time: 0.013853 ms
sigmoid_fp16x2                 mean time: 0.013854 ms
sigmoid_fp16x8                 mean time: 0.013738 ms
sigmoid_fp16x8_packed          mean time: 0.010093 ms
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.016591 ms
sigmoid                        mean time: 0.025278 ms
sigmoid_fp32x4                 mean time: 0.015882 ms
torch                          mean time: 0.014177 ms
sigmoid_half                   mean time: 0.029033 ms
sigmoid_fp16x2                 mean time: 0.014602 ms
sigmoid_fp16x8                 mean time: 0.028388 ms
sigmoid_fp16x8_packed          mean time: 0.013160 ms
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.037896 ms
sigmoid                        mean time: 0.061904 ms
sigmoid_fp32x4                 mean time: 0.037510 ms
torch                          mean time: 0.021204 ms
sigmoid_half                   mean time: 0.059834 ms
sigmoid_fp16x2                 mean time: 0.030586 ms
sigmoid_fp16x8                 mean time: 0.043520 ms
sigmoid_fp16x8_packed          mean time: 0.018923 ms
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.023421 ms
sigmoid                        mean time: 0.028850 ms
sigmoid_fp32x4                 mean time: 0.019954 ms
torch                          mean time: 0.013055 ms
sigmoid_half                   mean time: 0.023212 ms
sigmoid_fp16x2                 mean time: 0.014874 ms
sigmoid_fp16x8                 mean time: 0.028208 ms
sigmoid_fp16x8_packed          mean time: 0.010952 ms
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.039660 ms
sigmoid                        mean time: 0.061244 ms
sigmoid_fp32x4                 mean time: 0.037588 ms
torch                          mean time: 0.026857 ms
sigmoid_half                   mean time: 0.060812 ms
sigmoid_fp16x2                 mean time: 0.031650 ms
sigmoid_fp16x8                 mean time: 0.042678 ms
sigmoid_fp16x8_packed          mean time: 0.019441 ms
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.207171 ms
sigmoid                        mean time: 0.214822 ms
sigmoid_fp32x4                 mean time: 0.216711 ms
torch                          mean time: 0.062839 ms
sigmoid_half                   mean time: 0.149914 ms
sigmoid_fp16x2                 mean time: 0.060020 ms
sigmoid_fp16x8                 mean time: 0.108805 ms
sigmoid_fp16x8_packed          mean time: 0.032530 ms
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.043458 ms
sigmoid                        mean time: 0.072198 ms
sigmoid_fp32x4                 mean time: 0.039296 ms
torch                          mean time: 0.043283 ms
sigmoid_half                   mean time: 0.054822 ms
sigmoid_fp16x2                 mean time: 0.030962 ms
sigmoid_fp16x8                 mean time: 0.048095 ms
sigmoid_fp16x8_packed          mean time: 0.014379 ms
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.215525 ms
sigmoid                        mean time: 0.228766 ms
sigmoid_fp32x4                 mean time: 0.215321 ms
torch                          mean time: 0.041380 ms
sigmoid_half                   mean time: 0.145101 ms
sigmoid_fp16x2                 mean time: 0.063049 ms
sigmoid_fp16x8                 mean time: 0.128874 ms
sigmoid_fp16x8_packed          mean time: 0.032434 ms
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.403445 ms
sigmoid                        mean time: 0.443679 ms
sigmoid_fp32x4                 mean time: 0.410460 ms
torch                          mean time: 0.214407 ms
sigmoid_half                   mean time: 0.386082 ms
sigmoid_fp16x2                 mean time: 0.221343 ms
sigmoid_fp16x8                 mean time: 0.227966 ms
sigmoid_fp16x8_packed          mean time: 0.213118 ms
```
