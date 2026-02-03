# Dot Product

## 说明

涵盖了dot_product各版本kernel

- [x] dot_product fp32/fp16 版
- [x] dot_product_fp32x4(fp32向量化)
- [x] dot_product_fp16x2(fp16向量化)
- [x] dot_product_fp16x8(fp16向量化, packed r/w)
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
torch                          mean time: 0.021727 ms
dot_product                    mean time: 0.029790 ms
dot_product_fp32x4             mean time: 0.023043 ms
torch                          mean time: 0.019595 ms
dot_product                    mean time: 0.033884 ms
dot_product_fp16x2             mean time: 0.027985 ms
dot_product_fp16x8_packed      mean time: 0.028080 ms
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.024139 ms
dot_product                    mean time: 0.026223 ms
dot_product_fp32x4             mean time: 0.023965 ms
torch                          mean time: 0.021862 ms
dot_product                    mean time: 0.030724 ms
dot_product_fp16x2             mean time: 0.034279 ms
dot_product_fp16x8_packed      mean time: 0.033783 ms
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.027692 ms
dot_product                    mean time: 0.028867 ms
dot_product_fp32x4             mean time: 0.028988 ms
torch                          mean time: 0.027175 ms
dot_product                    mean time: 0.031102 ms
dot_product_fp16x2             mean time: 0.035025 ms
dot_product_fp16x8_packed      mean time: 0.039510 ms
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.032513 ms
dot_product                    mean time: 0.032702 ms
dot_product_fp32x4             mean time: 0.023970 ms
torch                          mean time: 0.019796 ms
dot_product                    mean time: 0.031500 ms
dot_product_fp16x2             mean time: 0.030058 ms
dot_product_fp16x8_packed      mean time: 0.030487 ms
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.029508 ms
dot_product                    mean time: 0.035381 ms
dot_product_fp32x4             mean time: 0.032802 ms
torch                          mean time: 0.034379 ms
dot_product                    mean time: 0.030316 ms
dot_product_fp16x2             mean time: 0.029050 ms
dot_product_fp16x8_packed      mean time: 0.030802 ms
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.190342 ms
dot_product                    mean time: 0.188644 ms
dot_product_fp32x4             mean time: 0.188322 ms
torch                          mean time: 0.035902 ms
dot_product                    mean time: 0.051770 ms
dot_product_fp16x2             mean time: 0.041512 ms
dot_product_fp16x8_packed      mean time: 0.033171 ms
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.028468 ms
dot_product                    mean time: 0.030862 ms
dot_product_fp32x4             mean time: 0.028530 ms
torch                          mean time: 0.031013 ms
dot_product                    mean time: 0.055556 ms
dot_product_fp16x2             mean time: 0.073013 ms
dot_product_fp16x8_packed      mean time: 0.037091 ms
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.190466 ms
dot_product                    mean time: 0.189381 ms
dot_product_fp32x4             mean time: 0.189973 ms
torch                          mean time: 0.032947 ms
dot_product                    mean time: 0.048509 ms
dot_product_fp16x2             mean time: 0.037139 ms
dot_product_fp16x8_packed      mean time: 0.032132 ms
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.435416 ms
dot_product                    mean time: 0.428888 ms
dot_product_fp32x4             mean time: 0.388114 ms
torch                          mean time: 0.195464 ms
dot_product                    mean time: 0.213447 ms
dot_product_fp16x2             mean time: 0.198001 ms
dot_product_fp16x8_packed      mean time: 0.197020 ms
```
