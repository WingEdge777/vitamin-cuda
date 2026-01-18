# GELU

## 说明

gelu kernel

- [x] gelu fp32/fp16 版
- [x] gelu_fp16x2(fp16向量化)
- [x] gelu_fp16x8(fp16向量化)
- [x] gelu_fp16x8(fp16向量化，packed r/w, half2 近两倍提升)
- [x] pytorch op bindings && diff check

## 测试

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

这里 pytorch 的 GELU 实现没有提供预分配 output buffer 的版本，所以对比结果不是很公平，理论 pytorch 自带劣势。但在多次循环下，pytorch allocator 开销可以忽略，实际测试 pytorch 跑得飞快，原因待研究

### 输出

```bash
####################################################################################################
n: 1024, m: 1024
torch                          mean time: 0.014492 ms
gelu                           mean time: 0.062652 ms
gelu_fp32x4                    mean time: 0.063183 ms
torch                          mean time: 0.010156 ms
gelu_half                      mean time: 0.017129 ms
gelu_fp16x2                    mean time: 0.016208 ms
gelu_fp16x8                    mean time: 0.024441 ms
gelu_fp16x8_packed             mean time: 0.022402 ms
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.016462 ms
gelu                           mean time: 0.121826 ms
gelu_fp32x4                    mean time: 0.118058 ms
torch                          mean time: 0.013990 ms
gelu_half                      mean time: 0.024989 ms
gelu_fp16x2                    mean time: 0.019352 ms
gelu_fp16x8                    mean time: 0.023147 ms
gelu_fp16x8_packed             mean time: 0.014866 ms
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.028529 ms
gelu                           mean time: 0.225693 ms
gelu_fp32x4                    mean time: 0.219614 ms
torch                          mean time: 0.027090 ms
gelu_half                      mean time: 0.059549 ms
gelu_fp16x2                    mean time: 0.038520 ms
gelu_fp16x8                    mean time: 0.045309 ms
gelu_fp16x8_packed             mean time: 0.030017 ms
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.033798 ms
gelu                           mean time: 0.106670 ms
gelu_fp32x4                    mean time: 0.102270 ms
torch                          mean time: 0.015839 ms
gelu_half                      mean time: 0.027566 ms
gelu_fp16x2                    mean time: 0.022942 ms
gelu_fp16x8                    mean time: 0.022374 ms
gelu_fp16x8_packed             mean time: 0.015834 ms
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.027197 ms
gelu                           mean time: 0.234069 ms
gelu_fp32x4                    mean time: 0.214255 ms
torch                          mean time: 0.028465 ms
gelu_half                      mean time: 0.057998 ms
gelu_fp16x2                    mean time: 0.040806 ms
gelu_fp16x8                    mean time: 0.049537 ms
gelu_fp16x8_packed             mean time: 0.027971 ms
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.211726 ms
gelu                           mean time: 0.491639 ms
gelu_fp32x4                    mean time: 0.559556 ms
torch                          mean time: 0.065631 ms
gelu_half                      mean time: 0.147252 ms
gelu_fp16x2                    mean time: 0.098261 ms
gelu_fp16x8                    mean time: 0.120161 ms
gelu_fp16x8_packed             mean time: 0.102666 ms
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.081989 ms
gelu                           mean time: 0.267879 ms
gelu_fp32x4                    mean time: 0.229424 ms
torch                          mean time: 0.025621 ms
gelu_half                      mean time: 0.063478 ms
gelu_fp16x2                    mean time: 0.040039 ms
gelu_fp16x8                    mean time: 0.047124 ms
gelu_fp16x8_packed             mean time: 0.038139 ms
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.272513 ms
gelu                           mean time: 0.529609 ms
gelu_fp32x4                    mean time: 0.490938 ms
torch                          mean time: 0.067841 ms
gelu_half                      mean time: 0.141394 ms
gelu_fp16x2                    mean time: 0.090334 ms
gelu_fp16x8                    mean time: 0.108358 ms
gelu_fp16x8_packed             mean time: 0.062593 ms
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.523081 ms
gelu                           mean time: 0.951924 ms
gelu_fp32x4                    mean time: 0.880450 ms
torch                          mean time: 0.212929 ms
gelu_half                      mean time: 0.354050 ms
gelu_fp16x2                    mean time: 0.234420 ms
gelu_fp16x8                    mean time: 0.218763 ms
gelu_fp16x8_packed             mean time: 0.214239 ms
```
