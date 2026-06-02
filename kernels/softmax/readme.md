# Softmax

## Overview

High-performance safe online softmax kernels with multiple strategies for different row sizes.

- [x] One-pass (single-row-per-block, register-resident)
  - [x] Safe online softmax — FP32 / FP16
  - [x] Safe online softmax — FP32 vectorized (×4)
  - [x] Safe online softmax — FP16 vectorized (×8, pure register, packed r/w)
  - [x] Safe online softmax **medium** — FP16 vectorized (moderate register + SMEM, packed r/w)
  - [x] Safe online softmax **extreme** — FP16 vectorized (max register + SMEM, packed r/w)
- [x] Two-pass (for large row sizes beyond single-block capacity)
  - [x] Safe online softmax **arbitrary** — FP16 vectorized (max register + SMEM, packed r/w)
  - [x] Safe online softmax **split-k** — FP16 vectorized (max register + SMEM, packed r/w)
- [x] PyTorch op binding & correctness check

## Build & Test

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### PTX Output

```yaml
ptxas info    : 28 bytes gmem
ptxas info    : Compiling entry function '_Z18softmax_grid_pass2ILi256ELi16EEvP6__halfS1_PfS2_i' for 'sm_120'
ptxas info    : Function properties for _Z18softmax_grid_pass2ILi256ELi16EEvP6__halfS1_PfS2_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 40 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 84.659 ms
ptxas info    : Compiling entry function '_Z18softmax_grid_pass1ILi256ELi16EEvP6__halfPfS2_i' for 'sm_120'
ptxas info    : Function properties for _Z18softmax_grid_pass1ILi256ELi16EEvP6__halfPfS2_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 79 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 21.369 ms
ptxas info    : Compiling entry function '_Z24softmax_arbitrary_kernelILi256EEvP6__halfS1_i' for 'sm_120'
ptxas info    : Function properties for _Z24softmax_arbitrary_kernelILi256EEvP6__halfS1_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 36 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 12.731 ms
ptxas info    : Compiling entry function '_Z22softmax_onepass_kernelILi256ELi32ELi24ELi1EEvP6__halfS1_i' for 'sm_120'
ptxas info    : Function properties for _Z22softmax_onepass_kernelILi256ELi32ELi24ELi1EEvP6__halfS1_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 250 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 176.764 ms
ptxas info    : Compiling entry function '_Z22softmax_onepass_kernelILi256ELi8ELi8ELi3EEvP6__halfS1_i' for 'sm_120'
ptxas info    : Function properties for _Z22softmax_onepass_kernelILi256ELi8ELi8ELi3EEvP6__halfS1_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 80 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 37.731 ms
ptxas info    : Compiling entry function '_Z28softmax_fp16x8_packed_kernelILi256EEvP6__halfS1_i' for 'sm_120'
ptxas info    : Function properties for _Z28softmax_fp16x8_packed_kernelILi256EEvP6__halfS1_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 95 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 38.098 ms
ptxas info    : Compiling entry function '_Z21softmax_fp32x4_kernelILi256EEvPfS0_i' for 'sm_120'
ptxas info    : Function properties for _Z21softmax_fp32x4_kernelILi256EEvPfS0_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 55 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 10.820 ms
ptxas info    : Compiling entry function '_Z14softmax_kernelILi256E6__halfEvPT0_S2_i' for 'sm_120'
ptxas info    : Function properties for _Z14softmax_kernelILi256E6__halfEvPT0_S2_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 90 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 20.868 ms
ptxas info    : Compiling entry function '_Z14softmax_kernelILi256EfEvPT0_S1_i' for 'sm_120'
ptxas info    : Function properties for _Z14softmax_kernelILi256EfEvPT0_S1_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 86 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 17.167 ms
```

### Benchmark Results

#### Large Row Sizes

```yaml
####################################################################################################
bs: 4, hidden_size: 16384
torch                          mean time: 0.010752 ms
softmax_medium                 mean time: 0.009040 ms, speedup: 1.19
softmax_extreme                mean time: 0.013120 ms, speedup: 0.82
softmax_arbitrary              mean time: 0.009216 ms, speedup: 1.17
softmax_splitk                 mean time: 0.014928 ms, speedup: 0.72
####################################################################################################
bs: 4, hidden_size: 32768
torch                          mean time: 0.013216 ms
softmax_medium                 mean time: 0.011072 ms, speedup: 1.19
softmax_extreme                mean time: 0.014928 ms, speedup: 0.89
softmax_arbitrary              mean time: 0.012992 ms, speedup: 1.02
softmax_splitk                 mean time: 0.017328 ms, speedup: 0.76
####################################################################################################
bs: 4, hidden_size: 65536
torch                          mean time: 0.019792 ms
softmax_extreme                mean time: 0.017712 ms, speedup: 1.12
softmax_arbitrary              mean time: 0.021504 ms, speedup: 0.92
softmax_splitk                 mean time: 0.017872 ms, speedup: 1.11
####################################################################################################
bs: 4, hidden_size: 114688
torch                          mean time: 0.029744 ms
softmax_extreme                mean time: 0.021392 ms, speedup: 1.39
softmax_arbitrary              mean time: 0.031872 ms, speedup: 0.93
softmax_splitk                 mean time: 0.020512 ms, speedup: 1.45
####################################################################################################
bs: 4, hidden_size: 262144
torch                          mean time: 0.061904 ms
softmax_arbitrary              mean time: 0.064176 ms, speedup: 0.96
softmax_splitk                 mean time: 0.023168 ms, speedup: 2.67
####################################################################################################
bs: 4, hidden_size: 1048576
torch                          mean time: 0.202112 ms
softmax_arbitrary              mean time: 0.233776 ms, speedup: 0.86
softmax_splitk                 mean time: 0.054720 ms, speedup: 3.69
####################################################################################################
bs: 4, hidden_size: 8388608
torch                          mean time: 1.730880 ms
softmax_arbitrary              mean time: 2.568016 ms, speedup: 0.67
softmax_splitk                 mean time: 0.603280 ms, speedup: 2.87
####################################################################################################
bs: 4, hidden_size: 33554432
torch                          mean time: 6.900256 ms
softmax_arbitrary              mean time: 10.154320 ms, speedup: 0.68
softmax_splitk                 mean time: 2.340544 ms, speedup: 2.95
```

#### Small Row Sizes

```yaml
ninja: no work to do.
####################################################################################################
bs: 128, hidden_size: 1024
torch                          mean time: 0.011968 ms
softmax_fp16                   mean time: 0.013216 ms, speedup: 0.91
softmax_fp16x8_packed          mean time: 0.012608 ms, speedup: 0.95
softmax_medium                 mean time: 0.010784 ms, speedup: 1.11
softmax_extreme                mean time: 0.035936 ms, speedup: 0.33
softmax_arbitrary              mean time: 0.008704 ms, speedup: 1.37
####################################################################################################
bs: 128, hidden_size: 2048
torch                          mean time: 0.016896 ms
softmax_fp16                   mean time: 0.015680 ms, speedup: 1.08
softmax_fp16x8_packed          mean time: 0.013216 ms, speedup: 1.28
softmax_medium                 mean time: 0.011296 ms, speedup: 1.50
softmax_extreme                mean time: 0.038096 ms, speedup: 0.44
softmax_arbitrary              mean time: 0.008880 ms, speedup: 1.90
####################################################################################################
bs: 128, hidden_size: 4096
torch                          mean time: 0.019776 ms
softmax_fp16                   mean time: 0.019328 ms, speedup: 1.02
softmax_fp16x8_packed          mean time: 0.015968 ms, speedup: 1.24
softmax_medium                 mean time: 0.012912 ms, speedup: 1.53
softmax_extreme                mean time: 0.039920 ms, speedup: 0.50
softmax_arbitrary              mean time: 0.010832 ms, speedup: 1.83
####################################################################################################
bs: 128, hidden_size: 8192
torch                          mean time: 0.029472 ms
softmax_fp16                   mean time: 0.027776 ms, speedup: 1.06
softmax_fp16x8_packed          mean time: 0.018192 ms, speedup: 1.62
softmax_medium                 mean time: 0.015424 ms, speedup: 1.91
softmax_extreme                mean time: 0.041616 ms, speedup: 0.71
softmax_arbitrary              mean time: 0.014848 ms, speedup: 1.98
####################################################################################################
bs: 256, hidden_size: 1024
torch                          mean time: 0.013280 ms
softmax_fp16                   mean time: 0.019040 ms, speedup: 0.70
softmax_fp16x8_packed          mean time: 0.016832 ms, speedup: 0.79
softmax_medium                 mean time: 0.014528 ms, speedup: 0.91
softmax_extreme                mean time: 0.064000 ms, speedup: 0.21
softmax_arbitrary              mean time: 0.009536 ms, speedup: 1.39
####################################################################################################
bs: 256, hidden_size: 2048
torch                          mean time: 0.019424 ms
softmax_fp16                   mean time: 0.022848 ms, speedup: 0.85
softmax_fp16x8_packed          mean time: 0.019552 ms, speedup: 0.99
softmax_medium                 mean time: 0.017184 ms, speedup: 1.13
softmax_extreme                mean time: 0.066656 ms, speedup: 0.29
softmax_arbitrary              mean time: 0.012016 ms, speedup: 1.62
####################################################################################################
bs: 256, hidden_size: 4096
torch                          mean time: 0.035696 ms
softmax_fp16                   mean time: 0.029792 ms, speedup: 1.20
softmax_fp16x8_packed          mean time: 0.023616 ms, speedup: 1.51
softmax_medium                 mean time: 0.017600 ms, speedup: 2.03
softmax_extreme                mean time: 0.069232 ms, speedup: 0.52
softmax_arbitrary              mean time: 0.014720 ms, speedup: 2.42
####################################################################################################
bs: 256, hidden_size: 8192
torch                          mean time: 0.052144 ms
softmax_fp16                   mean time: 0.045152 ms, speedup: 1.15
softmax_fp16x8_packed          mean time: 0.028864 ms, speedup: 1.81
softmax_medium                 mean time: 0.023712 ms, speedup: 2.20
softmax_extreme                mean time: 0.073760 ms, speedup: 0.71
softmax_arbitrary              mean time: 0.025648 ms, speedup: 2.03
####################################################################################################
bs: 1024, hidden_size: 1024
torch                          mean time: 0.023488 ms
softmax_fp16                   mean time: 0.054608 ms, speedup: 0.43
softmax_fp16x8_packed          mean time: 0.047728 ms, speedup: 0.49
softmax_medium                 mean time: 0.037696 ms, speedup: 0.62
softmax_extreme                mean time: 0.231200 ms, speedup: 0.10
softmax_arbitrary              mean time: 0.017600 ms, speedup: 1.33
####################################################################################################
bs: 1024, hidden_size: 2048
torch                          mean time: 0.059168 ms
softmax_fp16                   mean time: 0.066816 ms, speedup: 0.89
softmax_fp16x8_packed          mean time: 0.054352 ms, speedup: 1.09
softmax_medium                 mean time: 0.043840 ms, speedup: 1.35
softmax_extreme                mean time: 0.249744 ms, speedup: 0.24
softmax_arbitrary              mean time: 0.024720 ms, speedup: 2.39
####################################################################################################
bs: 1024, hidden_size: 4096
torch                          mean time: 0.102240 ms
softmax_fp16                   mean time: 0.089808 ms, speedup: 1.14
softmax_fp16x8_packed          mean time: 0.076880 ms, speedup: 1.33
softmax_medium                 mean time: 0.051488 ms, speedup: 1.99
softmax_extreme                mean time: 0.251840 ms, speedup: 0.41
softmax_arbitrary              mean time: 0.053200 ms, speedup: 1.92
####################################################################################################
bs: 1024, hidden_size: 8192
torch                          mean time: 0.168304 ms
softmax_fp16                   mean time: 0.139968 ms, speedup: 1.20
softmax_fp16x8_packed          mean time: 0.111104 ms, speedup: 1.51
softmax_medium                 mean time: 0.105824 ms, speedup: 1.59
softmax_extreme                mean time: 0.263888 ms, speedup: 0.64
softmax_arbitrary              mean time: 0.105680 ms, speedup: 1.59
####################################################################################################
bs: 2048, hidden_size: 1024
torch                          mean time: 0.035936 ms
softmax_fp16                   mean time: 0.101328 ms, speedup: 0.35
softmax_fp16x8_packed          mean time: 0.086832 ms, speedup: 0.41
softmax_medium                 mean time: 0.070768 ms, speedup: 0.51
softmax_extreme                mean time: 0.444448 ms, speedup: 0.08
softmax_arbitrary              mean time: 0.030448 ms, speedup: 1.18
####################################################################################################
bs: 2048, hidden_size: 2048
torch                          mean time: 0.092544 ms
softmax_fp16                   mean time: 0.121664 ms, speedup: 0.76
softmax_fp16x8_packed          mean time: 0.098736 ms, speedup: 0.94
softmax_medium                 mean time: 0.077184 ms, speedup: 1.20
softmax_extreme                mean time: 0.469632 ms, speedup: 0.20
softmax_arbitrary              mean time: 0.054768 ms, speedup: 1.69
####################################################################################################
bs: 2048, hidden_size: 4096
torch                          mean time: 0.191808 ms
softmax_fp16                   mean time: 0.161408 ms, speedup: 1.19
softmax_fp16x8_packed          mean time: 0.137248 ms, speedup: 1.40
softmax_medium                 mean time: 0.108080 ms, speedup: 1.77
softmax_extreme                mean time: 0.477984 ms, speedup: 0.40
softmax_arbitrary              mean time: 0.105664 ms, speedup: 1.82
####################################################################################################
bs: 2048, hidden_size: 8192
torch                          mean time: 0.322592 ms
softmax_fp16                   mean time: 0.249664 ms, speedup: 1.29
softmax_fp16x8_packed          mean time: 0.210896 ms, speedup: 1.53
softmax_medium                 mean time: 0.204560 ms, speedup: 1.58
softmax_extreme                mean time: 0.491744 ms, speedup: 0.66
softmax_arbitrary              mean time: 0.210160 ms, speedup: 1.53
```
