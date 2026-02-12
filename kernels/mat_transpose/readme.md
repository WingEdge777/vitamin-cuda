# transpose

## 说明

transpose kernel

- [x] transpose_coalesced_read (input视角，合并读)
- [x] transpose_coalesced_write (output视角，合并写)
- [x] transpose_smem (共享内存缓存，块状读写)
- [x] transpose_smem_bcf (共享内存无冲突版)
- [x] transpose_smem_packed (共享内存缓存，float4向量化读写)
- [x] transpose_smem_packed (共享内存无冲突版，float4向量化读写)
- [x] pytorch op bindings && diff check

## 测试

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### 输出

```bash
####################################################################################################
n: 256, m: 256
torch                          mean time: 0.015936 ms
transpose_coalesced_read       mean time: 0.009707 ms, speedup: 1.64
transpose_coalesced_write      mean time: 0.011806 ms, speedup: 1.35
transpose_smem                 mean time: 0.009900 ms, speedup: 1.61
transpose_smem_bcf             mean time: 0.009187 ms, speedup: 1.73
transpose_smem_packed          mean time: 0.008747 ms, speedup: 1.82
transpose_smem_swizzled_packed mean time: 0.009859 ms, speedup: 1.62
####################################################################################################
n: 256, m: 512
torch                          mean time: 0.039022 ms
transpose_coalesced_read       mean time: 0.009810 ms, speedup: 3.98
transpose_coalesced_write      mean time: 0.013379 ms, speedup: 2.92
transpose_smem                 mean time: 0.013272 ms, speedup: 2.94
transpose_smem_bcf             mean time: 0.007031 ms, speedup: 5.55
transpose_smem_packed          mean time: 0.016290 ms, speedup: 2.40
transpose_smem_swizzled_packed mean time: 0.008407 ms, speedup: 4.64
####################################################################################################
n: 256, m: 1024
torch                          mean time: 0.029538 ms
transpose_coalesced_read       mean time: 0.016925 ms, speedup: 1.75
transpose_coalesced_write      mean time: 0.013754 ms, speedup: 2.15
transpose_smem                 mean time: 0.009535 ms, speedup: 3.10
transpose_smem_bcf             mean time: 0.015666 ms, speedup: 1.89
transpose_smem_packed          mean time: 0.007503 ms, speedup: 3.94
transpose_smem_swizzled_packed mean time: 0.007936 ms, speedup: 3.72
####################################################################################################
n: 256, m: 2048
torch                          mean time: 0.037301 ms
transpose_coalesced_read       mean time: 0.013804 ms, speedup: 2.70
transpose_coalesced_write      mean time: 0.011290 ms, speedup: 3.30
transpose_smem                 mean time: 0.019872 ms, speedup: 1.88
transpose_smem_bcf             mean time: 0.008322 ms, speedup: 4.48
transpose_smem_packed          mean time: 0.009289 ms, speedup: 4.02
transpose_smem_swizzled_packed mean time: 0.008837 ms, speedup: 4.22
####################################################################################################
n: 256, m: 4096
torch                          mean time: 0.061659 ms
transpose_coalesced_read       mean time: 0.018582 ms, speedup: 3.32
transpose_coalesced_write      mean time: 0.016348 ms, speedup: 3.77
transpose_smem                 mean time: 0.023329 ms, speedup: 2.64
transpose_smem_bcf             mean time: 0.011762 ms, speedup: 5.24
transpose_smem_packed          mean time: 0.019081 ms, speedup: 3.23
transpose_smem_swizzled_packed mean time: 0.011517 ms, speedup: 5.35
####################################################################################################
n: 256, m: 8192
torch                          mean time: 0.099356 ms
transpose_coalesced_read       mean time: 0.039415 ms, speedup: 2.52
transpose_coalesced_write      mean time: 0.035658 ms, speedup: 2.79
transpose_smem                 mean time: 0.051510 ms, speedup: 1.93
transpose_smem_bcf             mean time: 0.016975 ms, speedup: 5.85
transpose_smem_packed          mean time: 0.022649 ms, speedup: 4.39
transpose_smem_swizzled_packed mean time: 0.015410 ms, speedup: 6.45
####################################################################################################
n: 512, m: 256
torch                          mean time: 0.029188 ms
transpose_coalesced_read       mean time: 0.010946 ms, speedup: 2.67
transpose_coalesced_write      mean time: 0.008106 ms, speedup: 3.60
transpose_smem                 mean time: 0.012789 ms, speedup: 2.28
transpose_smem_bcf             mean time: 0.011662 ms, speedup: 2.50
transpose_smem_packed          mean time: 0.013594 ms, speedup: 2.15
transpose_smem_swizzled_packed mean time: 0.013146 ms, speedup: 2.22
####################################################################################################
n: 512, m: 512
torch                          mean time: 0.021600 ms
transpose_coalesced_read       mean time: 0.018999 ms, speedup: 1.14
transpose_coalesced_write      mean time: 0.013921 ms, speedup: 1.55
transpose_smem                 mean time: 0.015625 ms, speedup: 1.38
transpose_smem_bcf             mean time: 0.012267 ms, speedup: 1.76
transpose_smem_packed          mean time: 0.013524 ms, speedup: 1.60
transpose_smem_swizzled_packed mean time: 0.008335 ms, speedup: 2.59
####################################################################################################
n: 512, m: 1024
torch                          mean time: 0.044117 ms
transpose_coalesced_read       mean time: 0.012088 ms, speedup: 3.65
transpose_coalesced_write      mean time: 0.019107 ms, speedup: 2.31
transpose_smem                 mean time: 0.016888 ms, speedup: 2.61
transpose_smem_bcf             mean time: 0.009359 ms, speedup: 4.71
transpose_smem_packed          mean time: 0.009006 ms, speedup: 4.90
transpose_smem_swizzled_packed mean time: 0.009208 ms, speedup: 4.79
####################################################################################################
n: 512, m: 2048
torch                          mean time: 0.043792 ms
transpose_coalesced_read       mean time: 0.017015 ms, speedup: 2.57
transpose_coalesced_write      mean time: 0.016623 ms, speedup: 2.63
transpose_smem                 mean time: 0.024426 ms, speedup: 1.79
transpose_smem_bcf             mean time: 0.012252 ms, speedup: 3.57
transpose_smem_packed          mean time: 0.019959 ms, speedup: 2.19
transpose_smem_swizzled_packed mean time: 0.011462 ms, speedup: 3.82
####################################################################################################
n: 512, m: 4096
torch                          mean time: 0.096561 ms
transpose_coalesced_read       mean time: 0.041777 ms, speedup: 2.31
transpose_coalesced_write      mean time: 0.029948 ms, speedup: 3.22
transpose_smem                 mean time: 0.047850 ms, speedup: 2.02
transpose_smem_bcf             mean time: 0.015993 ms, speedup: 6.04
transpose_smem_packed          mean time: 0.016147 ms, speedup: 5.98
transpose_smem_swizzled_packed mean time: 0.021810 ms, speedup: 4.43
####################################################################################################
n: 512, m: 8192
torch                          mean time: 0.235638 ms
transpose_coalesced_read       mean time: 0.087566 ms, speedup: 2.69
transpose_coalesced_write      mean time: 0.081432 ms, speedup: 2.89
transpose_smem                 mean time: 0.121227 ms, speedup: 1.94
transpose_smem_bcf             mean time: 0.039738 ms, speedup: 5.93
transpose_smem_packed          mean time: 0.041985 ms, speedup: 5.61
transpose_smem_swizzled_packed mean time: 0.044858 ms, speedup: 5.25
####################################################################################################
n: 1024, m: 256
torch                          mean time: 0.043437 ms
transpose_coalesced_read       mean time: 0.012265 ms, speedup: 3.54
transpose_coalesced_write      mean time: 0.012784 ms, speedup: 3.40
transpose_smem                 mean time: 0.015446 ms, speedup: 2.81
transpose_smem_bcf             mean time: 0.015634 ms, speedup: 2.78
transpose_smem_packed          mean time: 0.017586 ms, speedup: 2.47
transpose_smem_swizzled_packed mean time: 0.008958 ms, speedup: 4.85
####################################################################################################
n: 1024, m: 512
torch                          mean time: 0.033235 ms
transpose_coalesced_read       mean time: 0.014024 ms, speedup: 2.37
transpose_coalesced_write      mean time: 0.010968 ms, speedup: 3.03
transpose_smem                 mean time: 0.013975 ms, speedup: 2.38
transpose_smem_bcf             mean time: 0.009227 ms, speedup: 3.60
transpose_smem_packed          mean time: 0.014660 ms, speedup: 2.27
transpose_smem_swizzled_packed mean time: 0.012648 ms, speedup: 2.63
####################################################################################################
n: 1024, m: 1024
torch                          mean time: 0.050235 ms
transpose_coalesced_read       mean time: 0.023619 ms, speedup: 2.13
transpose_coalesced_write      mean time: 0.016792 ms, speedup: 2.99
transpose_smem                 mean time: 0.031759 ms, speedup: 1.58
transpose_smem_bcf             mean time: 0.013282 ms, speedup: 3.78
transpose_smem_packed          mean time: 0.011877 ms, speedup: 4.23
transpose_smem_swizzled_packed mean time: 0.016005 ms, speedup: 3.14
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.100252 ms
transpose_coalesced_read       mean time: 0.041317 ms, speedup: 2.43
transpose_coalesced_write      mean time: 0.030270 ms, speedup: 3.31
transpose_smem                 mean time: 0.054474 ms, speedup: 1.84
transpose_smem_bcf             mean time: 0.017925 ms, speedup: 5.59
transpose_smem_packed          mean time: 0.014837 ms, speedup: 6.76
transpose_smem_swizzled_packed mean time: 0.025952 ms, speedup: 3.86
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.276840 ms
transpose_coalesced_read       mean time: 0.111108 ms, speedup: 2.49
transpose_coalesced_write      mean time: 0.097441 ms, speedup: 2.84
transpose_smem                 mean time: 0.133131 ms, speedup: 2.08
transpose_smem_bcf             mean time: 0.041056 ms, speedup: 6.74
transpose_smem_packed          mean time: 0.043916 ms, speedup: 6.30
transpose_smem_swizzled_packed mean time: 0.041213 ms, speedup: 6.72
####################################################################################################
n: 1024, m: 8192
torch                          mean time: 0.585723 ms
transpose_coalesced_read       mean time: 0.258797 ms, speedup: 2.26
transpose_coalesced_write      mean time: 0.262633 ms, speedup: 2.23
transpose_smem                 mean time: 0.251555 ms, speedup: 2.33
transpose_smem_bcf             mean time: 0.228486 ms, speedup: 2.56
transpose_smem_packed          mean time: 0.234188 ms, speedup: 2.50
transpose_smem_swizzled_packed mean time: 0.228353 ms, speedup: 2.56
####################################################################################################
n: 2048, m: 256
torch                          mean time: 0.028689 ms
transpose_coalesced_read       mean time: 0.011855 ms, speedup: 2.42
transpose_coalesced_write      mean time: 0.019980 ms, speedup: 1.44
transpose_smem                 mean time: 0.013028 ms, speedup: 2.20
transpose_smem_bcf             mean time: 0.010366 ms, speedup: 2.77
transpose_smem_packed          mean time: 0.021813 ms, speedup: 1.32
transpose_smem_swizzled_packed mean time: 0.009813 ms, speedup: 2.92
####################################################################################################
n: 2048, m: 512
torch                          mean time: 0.049766 ms
transpose_coalesced_read       mean time: 0.016454 ms, speedup: 3.02
transpose_coalesced_write      mean time: 0.015131 ms, speedup: 3.29
transpose_smem                 mean time: 0.029816 ms, speedup: 1.67
transpose_smem_bcf             mean time: 0.010639 ms, speedup: 4.68
transpose_smem_packed          mean time: 0.010791 ms, speedup: 4.61
transpose_smem_swizzled_packed mean time: 0.016965 ms, speedup: 2.93
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.090050 ms
transpose_coalesced_read       mean time: 0.035530 ms, speedup: 2.53
transpose_coalesced_write      mean time: 0.028418 ms, speedup: 3.17
transpose_smem                 mean time: 0.045907 ms, speedup: 1.96
transpose_smem_bcf             mean time: 0.015398 ms, speedup: 5.85
transpose_smem_packed          mean time: 0.017894 ms, speedup: 5.03
transpose_smem_swizzled_packed mean time: 0.014367 ms, speedup: 6.27
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.240581 ms
transpose_coalesced_read       mean time: 0.071583 ms, speedup: 3.36
transpose_coalesced_write      mean time: 0.070640 ms, speedup: 3.41
transpose_smem                 mean time: 0.107670 ms, speedup: 2.23
transpose_smem_bcf             mean time: 0.035163 ms, speedup: 6.84
transpose_smem_packed          mean time: 0.035360 ms, speedup: 6.80
transpose_smem_swizzled_packed mean time: 0.031067 ms, speedup: 7.74
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.571469 ms
transpose_coalesced_read       mean time: 0.264726 ms, speedup: 2.16
transpose_coalesced_write      mean time: 0.278389 ms, speedup: 2.05
transpose_smem                 mean time: 0.232085 ms, speedup: 2.46
transpose_smem_bcf             mean time: 0.221696 ms, speedup: 2.58
transpose_smem_packed          mean time: 0.227052 ms, speedup: 2.52
transpose_smem_swizzled_packed mean time: 0.221229 ms, speedup: 2.58
####################################################################################################
n: 2048, m: 8192
torch                          mean time: 1.176325 ms
transpose_coalesced_read       mean time: 0.512921 ms, speedup: 2.29
transpose_coalesced_write      mean time: 0.483430 ms, speedup: 2.43
transpose_smem                 mean time: 0.475731 ms, speedup: 2.47
transpose_smem_bcf             mean time: 0.451156 ms, speedup: 2.61
transpose_smem_packed          mean time: 0.451417 ms, speedup: 2.61
transpose_smem_swizzled_packed mean time: 0.444898 ms, speedup: 2.64
####################################################################################################
n: 4096, m: 256
torch                          mean time: 0.048857 ms
transpose_coalesced_read       mean time: 0.022841 ms, speedup: 2.14
transpose_coalesced_write      mean time: 0.014784 ms, speedup: 3.30
transpose_smem                 mean time: 0.022072 ms, speedup: 2.21
transpose_smem_bcf             mean time: 0.010041 ms, speedup: 4.87
transpose_smem_packed          mean time: 0.009692 ms, speedup: 5.04
transpose_smem_swizzled_packed mean time: 0.011929 ms, speedup: 4.10
####################################################################################################
n: 4096, m: 512
torch                          mean time: 0.097931 ms
transpose_coalesced_read       mean time: 0.038015 ms, speedup: 2.58
transpose_coalesced_write      mean time: 0.034128 ms, speedup: 2.87
transpose_smem                 mean time: 0.050574 ms, speedup: 1.94
transpose_smem_bcf             mean time: 0.015633 ms, speedup: 6.26
transpose_smem_packed          mean time: 0.020512 ms, speedup: 4.77
transpose_smem_swizzled_packed mean time: 0.015081 ms, speedup: 6.49
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.249520 ms
transpose_coalesced_read       mean time: 0.078372 ms, speedup: 3.18
transpose_coalesced_write      mean time: 0.082541 ms, speedup: 3.02
transpose_smem                 mean time: 0.117656 ms, speedup: 2.12
transpose_smem_bcf             mean time: 0.034904 ms, speedup: 7.15
transpose_smem_packed          mean time: 0.029064 ms, speedup: 8.59
transpose_smem_swizzled_packed mean time: 0.031473 ms, speedup: 7.93
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.650237 ms
transpose_coalesced_read       mean time: 0.336431 ms, speedup: 1.93
transpose_coalesced_write      mean time: 0.278399 ms, speedup: 2.34
transpose_smem                 mean time: 0.260425 ms, speedup: 2.50
transpose_smem_bcf             mean time: 0.221683 ms, speedup: 2.93
transpose_smem_packed          mean time: 0.220976 ms, speedup: 2.94
transpose_smem_swizzled_packed mean time: 0.213474 ms, speedup: 3.05
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 1.242258 ms
transpose_coalesced_read       mean time: 0.552140 ms, speedup: 2.25
transpose_coalesced_write      mean time: 0.520630 ms, speedup: 2.39
transpose_smem                 mean time: 0.465094 ms, speedup: 2.67
transpose_smem_bcf             mean time: 0.441158 ms, speedup: 2.82
transpose_smem_packed          mean time: 0.452693 ms, speedup: 2.74
transpose_smem_swizzled_packed mean time: 0.452228 ms, speedup: 2.75
####################################################################################################
n: 4096, m: 8192
torch                          mean time: 2.229296 ms
transpose_coalesced_read       mean time: 1.147599 ms, speedup: 1.94
transpose_coalesced_write      mean time: 0.923157 ms, speedup: 2.41
transpose_smem                 mean time: 0.889604 ms, speedup: 2.51
transpose_smem_bcf             mean time: 0.891839 ms, speedup: 2.50
transpose_smem_packed          mean time: 0.903418 ms, speedup: 2.47
transpose_smem_swizzled_packed mean time: 0.902854 ms, speedup: 2.47
####################################################################################################
n: 8192, m: 256
torch                          mean time: 0.099418 ms
transpose_coalesced_read       mean time: 0.029330 ms, speedup: 3.39
transpose_coalesced_write      mean time: 0.027233 ms, speedup: 3.65
transpose_smem                 mean time: 0.041025 ms, speedup: 2.42
transpose_smem_bcf             mean time: 0.017422 ms, speedup: 5.71
transpose_smem_packed          mean time: 0.017902 ms, speedup: 5.55
transpose_smem_swizzled_packed mean time: 0.021744 ms, speedup: 4.57
####################################################################################################
n: 8192, m: 512
torch                          mean time: 0.244327 ms
transpose_coalesced_read       mean time: 0.061740 ms, speedup: 3.96
transpose_coalesced_write      mean time: 0.055910 ms, speedup: 4.37
transpose_smem                 mean time: 0.102157 ms, speedup: 2.39
transpose_smem_bcf             mean time: 0.025202 ms, speedup: 9.69
transpose_smem_packed          mean time: 0.032813 ms, speedup: 7.45
transpose_smem_swizzled_packed mean time: 0.028582 ms, speedup: 8.55
####################################################################################################
n: 8192, m: 1024
torch                          mean time: 0.594003 ms
transpose_coalesced_read       mean time: 0.258670 ms, speedup: 2.30
transpose_coalesced_write      mean time: 0.257716 ms, speedup: 2.30
transpose_smem                 mean time: 0.234585 ms, speedup: 2.53
transpose_smem_bcf             mean time: 0.231517 ms, speedup: 2.57
transpose_smem_packed          mean time: 0.231619 ms, speedup: 2.56
transpose_smem_swizzled_packed mean time: 0.232089 ms, speedup: 2.56
####################################################################################################
n: 8192, m: 2048
torch                          mean time: 1.100863 ms
transpose_coalesced_read       mean time: 0.491941 ms, speedup: 2.24
transpose_coalesced_write      mean time: 0.483749 ms, speedup: 2.28
transpose_smem                 mean time: 0.414627 ms, speedup: 2.66
transpose_smem_bcf             mean time: 0.425189 ms, speedup: 2.59
transpose_smem_packed          mean time: 0.426199 ms, speedup: 2.58
transpose_smem_swizzled_packed mean time: 0.412782 ms, speedup: 2.67
####################################################################################################
n: 8192, m: 4096
torch                          mean time: 2.305065 ms
transpose_coalesced_read       mean time: 1.090572 ms, speedup: 2.11
transpose_coalesced_write      mean time: 1.011630 ms, speedup: 2.28
transpose_smem                 mean time: 0.859068 ms, speedup: 2.68
transpose_smem_bcf             mean time: 0.868001 ms, speedup: 2.66
transpose_smem_packed          mean time: 0.867811 ms, speedup: 2.66
transpose_smem_swizzled_packed mean time: 0.934817 ms, speedup: 2.47
####################################################################################################
n: 8192, m: 8192
torch                          mean time: 4.995466 ms
transpose_coalesced_read       mean time: 2.215831 ms, speedup: 2.25
transpose_coalesced_write      mean time: 1.882122 ms, speedup: 2.65
transpose_smem                 mean time: 1.835753 ms, speedup: 2.72
transpose_smem_bcf             mean time: 1.708558 ms, speedup: 2.92
transpose_smem_packed          mean time: 1.710325 ms, speedup: 2.92
transpose_smem_swizzled_packed mean time: 1.716763 ms, speedup: 2.91
```
