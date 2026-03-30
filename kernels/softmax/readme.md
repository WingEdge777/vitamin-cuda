# softmax

## 说明

softmax kernel

- [x] one pass
  - [x] safe online softmax fp32/fp16 版
  - [x] safe online softmax fp32x4 版 (fp32向量化)
  - [x] safe online softmax fp16x8 版 (fp16向量化, pure register，packed r/w)
  - [x] safe online softmax medium fp16 版 (fp16向量化, medium register+smem, packed r/w)
  - [x] safe online softmax extreme fp16 版 (fp16向量化, max register+smem, packed r/w)
- [x] two pass
  - [x] safe online softmax arbitrary fp16 版 (fp16向量化, max register+smem, packed r/w)
  - [x] safe online softmax split-k fp16 版 (fp16向量化, max register+smem, packed r/w)
- [x] pytorch op bindings && diff check

## 测试

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### ptx output

```bash
ptxas info    : 28 bytes gmem
ptxas info    : Compiling entry function '_Z18softmax_grid_pass2ILi256ELi16EEvP6__halfS1_PfS2_i' for 'sm_120'
ptxas info    : Function properties for _Z18softmax_grid_pass2ILi256ELi16EEvP6__halfS1_PfS2_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 40 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 46.738 ms
ptxas info    : Compiling entry function '_Z18softmax_grid_pass1ILi256ELi16EEvP6__halfPfS2_i' for 'sm_120'
ptxas info    : Function properties for _Z18softmax_grid_pass1ILi256ELi16EEvP6__halfPfS2_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 86 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 25.829 ms
ptxas info    : Compiling entry function '_Z24softmax_arbitrary_kernelILi256EEvP6__halfS1_i' for 'sm_120'
ptxas info    : Function properties for _Z24softmax_arbitrary_kernelILi256EEvP6__halfS1_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 36 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 17.098 ms
ptxas info    : Compiling entry function '_Z22softmax_onepass_kernelILi256ELi32ELi24ELi1EEvP6__halfS1_i' for 'sm_120'
ptxas info    : Function properties for _Z22softmax_onepass_kernelILi256ELi32ELi24ELi1EEvP6__halfS1_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 250 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 229.864 ms
ptxas info    : Compiling entry function '_Z22softmax_onepass_kernelILi256ELi8ELi8ELi3EEvP6__halfS1_i' for 'sm_120'
ptxas info    : Function properties for _Z22softmax_onepass_kernelILi256ELi8ELi8ELi3EEvP6__halfS1_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 80 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 30.446 ms
ptxas info    : Compiling entry function '_Z28softmax_fp16x8_packed_kernelILi256EEvP6__halfS1_i' for 'sm_120'
ptxas info    : Function properties for _Z28softmax_fp16x8_packed_kernelILi256EEvP6__halfS1_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 95 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 35.515 ms
ptxas info    : Compiling entry function '_Z21softmax_fp32x4_kernelILi256EEvPfS0_i' for 'sm_120'
ptxas info    : Function properties for _Z21softmax_fp32x4_kernelILi256EEvPfS0_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 55 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 9.114 ms
ptxas info    : Compiling entry function '_Z14softmax_kernelILi256E6__halfEvPT0_S2_i' for 'sm_120'
ptxas info    : Function properties for _Z14softmax_kernelILi256E6__halfEvPT0_S2_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 90 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 14.633 ms
ptxas info    : Compiling entry function '_Z14softmax_kernelILi256EfEvPT0_S1_i' for 'sm_120'
ptxas info    : Function properties for _Z14softmax_kernelILi256EfEvPT0_S1_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 86 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 13.406 ms
```

### 输出

#### large

```yaml
####################################################################################################
bs: 4, hidden_size: 16384
torch                          mean time: 0.012369 ms
softmax_medium                 mean time: 0.011401 ms, speedup: 1.08
softmax_extreme                mean time: 0.013799 ms, speedup: 0.90
softmax_arbitrary              mean time: 0.010522 ms, speedup: 1.18
softmax_splitk                 mean time: 0.016911 ms, speedup: 0.73
####################################################################################################
bs: 4, hidden_size: 32768
torch                          mean time: 0.033266 ms
softmax_medium                 mean time: 0.015141 ms, speedup: 2.20
softmax_extreme                mean time: 0.014320 ms, speedup: 2.32
softmax_arbitrary              mean time: 0.011193 ms, speedup: 2.97
softmax_splitk                 mean time: 0.020862 ms, speedup: 1.59
####################################################################################################
bs: 4, hidden_size: 65536
torch                          mean time: 0.016672 ms
softmax_extreme                mean time: 0.014532 ms, speedup: 1.15
softmax_arbitrary              mean time: 0.016860 ms, speedup: 0.99
softmax_splitk                 mean time: 0.046560 ms, speedup: 0.36
####################################################################################################
bs: 4, hidden_size: 114688
torch                          mean time: 0.023799 ms
softmax_extreme                mean time: 0.040429 ms, speedup: 0.59
softmax_arbitrary              mean time: 0.023173 ms, speedup: 1.03
softmax_splitk                 mean time: 0.020216 ms, speedup: 1.18
####################################################################################################
bs: 4, hidden_size: 262144
torch                          mean time: 0.043840 ms
softmax_arbitrary              mean time: 0.043765 ms, speedup: 1.00
softmax_splitk                 mean time: 0.045702 ms, speedup: 0.96
####################################################################################################
bs: 4, hidden_size: 1048576
torch                          mean time: 0.193523 ms
softmax_arbitrary              mean time: 0.158823 ms, speedup: 1.22
softmax_splitk                 mean time: 0.026112 ms, speedup: 7.41
####################################################################################################
bs: 4, hidden_size: 8388608
torch                          mean time: 2.036284 ms
softmax_arbitrary              mean time: 2.943902 ms, speedup: 0.69
softmax_splitk                 mean time: 0.689904 ms, speedup: 2.95
####################################################################################################
bs: 4, hidden_size: 33554432
torch                          mean time: 7.662010 ms
softmax_arbitrary              mean time: 10.514376 ms, speedup: 0.73
softmax_splitk                 mean time: 2.372789 ms, speedup: 3.23
```

#### small

```yaml
####################################################################################################
n: 256, m: 2048
torch                          mean time: 0.024114 ms
softmax                        mean time: 0.015870 ms
softmax_fp32x4                 mean time: 0.013032 ms
torch                          mean time: 0.014257 ms
softmax_fp16                   mean time: 0.014551 ms
softmax_fp16x8_packed          mean time: 0.013143 ms
####################################################################################################
n: 256, m: 4096
torch                          mean time: 0.031423 ms
softmax                        mean time: 0.029552 ms
softmax_fp32x4                 mean time: 0.028992 ms
torch                          mean time: 0.031730 ms
softmax_fp16                   mean time: 0.025955 ms
softmax_fp16x8_packed          mean time: 0.013955 ms
####################################################################################################
n: 256, m: 8192
torch                          mean time: 0.048221 ms
softmax                        mean time: 0.034189 ms
softmax_fp32x4                 mean time: 0.036261 ms
torch                          mean time: 0.055996 ms
softmax_fp16                   mean time: 0.030237 ms
softmax_fp16x8_packed          mean time: 0.019414 ms
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.032798 ms
softmax                        mean time: 0.043320 ms
softmax_fp32x4                 mean time: 0.043095 ms
torch                          mean time: 0.053495 ms
softmax_fp16                   mean time: 0.030184 ms
softmax_fp16x8_packed          mean time: 0.028999 ms
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.120721 ms
softmax                        mean time: 0.079462 ms
softmax_fp32x4                 mean time: 0.071995 ms
torch                          mean time: 0.132641 ms
softmax_fp16                   mean time: 0.050337 ms
softmax_fp16x8_packed          mean time: 0.027740 ms
####################################################################################################
n: 1024, m: 8192
torch                          mean time: 0.233507 ms
softmax                        mean time: 0.305752 ms
softmax_fp32x4                 mean time: 0.302634 ms
torch                          mean time: 0.154205 ms
softmax_fp16                   mean time: 0.150497 ms
softmax_fp16x8_packed          mean time: 0.106990 ms
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.041169 ms
softmax                        mean time: 0.090414 ms
softmax_fp32x4                 mean time: 0.072104 ms
torch                          mean time: 0.070662 ms
softmax_fp16                   mean time: 0.048812 ms
softmax_fp16x8_packed          mean time: 0.036938 ms
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.287007 ms
softmax                        mean time: 0.302273 ms
softmax_fp32x4                 mean time: 0.301171 ms
torch                          mean time: 0.210818 ms
softmax_fp16                   mean time: 0.122271 ms
softmax_fp16x8_packed          mean time: 0.126583 ms
####################################################################################################
n: 2048, m: 8192
torch                          mean time: 0.535019 ms
softmax                        mean time: 0.558522 ms
softmax_fp32x4                 mean time: 0.558709 ms
torch                          mean time: 0.433393 ms
softmax_fp16                   mean time: 0.289187 ms
softmax_fp16x8_packed          mean time: 0.330877 ms
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.326982 ms
softmax                        mean time: 0.320178 ms
softmax_fp32x4                 mean time: 0.303483 ms
torch                          mean time: 0.116638 ms
softmax_fp16                   mean time: 0.106106 ms
softmax_fp16x8_packed          mean time: 0.078162 ms
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.554743 ms
softmax                        mean time: 0.545286 ms
softmax_fp32x4                 mean time: 0.491609 ms
torch                          mean time: 0.478670 ms
softmax_fp16                   mean time: 0.255322 ms
softmax_fp16x8_packed          mean time: 0.253835 ms
####################################################################################################
n: 4096, m: 8192
torch                          mean time: 1.095348 ms
softmax                        mean time: 1.015634 ms
softmax_fp32x4                 mean time: 1.138277 ms
torch                          mean time: 0.942682 ms
softmax_fp16                   mean time: 0.590136 ms
softmax_fp16x8_packed          mean time: 0.583470 ms
```
