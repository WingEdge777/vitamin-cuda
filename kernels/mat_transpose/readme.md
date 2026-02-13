# transpose

## 说明

transpose kernel

- [x] transpose_coalesced_read (input视角，合并读)
- [x] transpose_coalesced_write (output视角，合并写)
- [x] transpose_smem (共享内存缓存，块状读写)
- [x] transpose_smem_bcf (共享内存无冲突版)
- [x] transpose_smem_packed_bcf (共享内存缓存，float4向量化读写)
- [x] transpose_smem_packed_bcf (共享内存无冲突版，float4向量化读写)
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
torch                          mean time: 0.018426 ms
transpose_coalesced_read       mean time: 0.017290 ms, speedup: 1.07
transpose_coalesced_write      mean time: 0.008075 ms, speedup: 2.28
transpose_smem                 mean time: 0.013925 ms, speedup: 1.32
transpose_smem_bcf             mean time: 0.012378 ms, speedup: 1.49
transpose_smem_packed_bcf      mean time: 0.014419 ms, speedup: 1.28
transpose_smem_swizzled_packed mean time: 0.010465 ms, speedup: 1.76
####################################################################################################
n: 256, m: 512
torch                          mean time: 0.035309 ms
transpose_coalesced_read       mean time: 0.008346 ms, speedup: 4.23
transpose_coalesced_write      mean time: 0.009452 ms, speedup: 3.74
transpose_smem                 mean time: 0.012590 ms, speedup: 2.80
transpose_smem_bcf             mean time: 0.015460 ms, speedup: 2.28
transpose_smem_packed_bcf      mean time: 0.007596 ms, speedup: 4.65
transpose_smem_swizzled_packed mean time: 0.008977 ms, speedup: 3.93
####################################################################################################
n: 256, m: 1024
torch                          mean time: 0.031019 ms
transpose_coalesced_read       mean time: 0.009799 ms, speedup: 3.17
transpose_coalesced_write      mean time: 0.010001 ms, speedup: 3.10
transpose_smem                 mean time: 0.010714 ms, speedup: 2.90
transpose_smem_bcf             mean time: 0.008325 ms, speedup: 3.73
transpose_smem_packed_bcf      mean time: 0.008099 ms, speedup: 3.83
transpose_smem_swizzled_packed mean time: 0.009331 ms, speedup: 3.32
####################################################################################################
n: 256, m: 2048
torch                          mean time: 0.036861 ms
transpose_coalesced_read       mean time: 0.012707 ms, speedup: 2.90
transpose_coalesced_write      mean time: 0.012325 ms, speedup: 2.99
transpose_smem                 mean time: 0.020456 ms, speedup: 1.80
transpose_smem_bcf             mean time: 0.009381 ms, speedup: 3.93
transpose_smem_packed_bcf      mean time: 0.009861 ms, speedup: 3.74
transpose_smem_swizzled_packed mean time: 0.008956 ms, speedup: 4.12
####################################################################################################
n: 256, m: 4096
torch                          mean time: 0.050317 ms
transpose_coalesced_read       mean time: 0.014935 ms, speedup: 3.37
transpose_coalesced_write      mean time: 0.014291 ms, speedup: 3.52
transpose_smem                 mean time: 0.024924 ms, speedup: 2.02
transpose_smem_bcf             mean time: 0.021057 ms, speedup: 2.39
transpose_smem_packed_bcf      mean time: 0.013025 ms, speedup: 3.86
transpose_smem_swizzled_packed mean time: 0.010983 ms, speedup: 4.58
####################################################################################################
n: 256, m: 8192
torch                          mean time: 0.080372 ms
transpose_coalesced_read       mean time: 0.031174 ms, speedup: 2.58
transpose_coalesced_write      mean time: 0.026320 ms, speedup: 3.05
transpose_smem                 mean time: 0.049795 ms, speedup: 1.61
transpose_smem_bcf             mean time: 0.014928 ms, speedup: 5.38
transpose_smem_packed_bcf      mean time: 0.013629 ms, speedup: 5.90
transpose_smem_swizzled_packed mean time: 0.014922 ms, speedup: 5.39
####################################################################################################
n: 512, m: 256
torch                          mean time: 0.031729 ms
transpose_coalesced_read       mean time: 0.008148 ms, speedup: 3.89
transpose_coalesced_write      mean time: 0.007499 ms, speedup: 4.23
transpose_smem                 mean time: 0.009210 ms, speedup: 3.44
transpose_smem_bcf             mean time: 0.009350 ms, speedup: 3.39
transpose_smem_packed_bcf      mean time: 0.012244 ms, speedup: 2.59
transpose_smem_swizzled_packed mean time: 0.009375 ms, speedup: 3.38
####################################################################################################
n: 512, m: 512
torch                          mean time: 0.023886 ms
transpose_coalesced_read       mean time: 0.015001 ms, speedup: 1.59
transpose_coalesced_write      mean time: 0.009618 ms, speedup: 2.48
transpose_smem                 mean time: 0.019231 ms, speedup: 1.24
transpose_smem_bcf             mean time: 0.008236 ms, speedup: 2.90
transpose_smem_packed_bcf      mean time: 0.008356 ms, speedup: 2.86
transpose_smem_swizzled_packed mean time: 0.007701 ms, speedup: 3.10
####################################################################################################
n: 512, m: 1024
torch                          mean time: 0.025825 ms
transpose_coalesced_read       mean time: 0.010915 ms, speedup: 2.37
transpose_coalesced_write      mean time: 0.011014 ms, speedup: 2.34
transpose_smem                 mean time: 0.012370 ms, speedup: 2.09
transpose_smem_bcf             mean time: 0.008914 ms, speedup: 2.90
transpose_smem_packed_bcf      mean time: 0.009180 ms, speedup: 2.81
transpose_smem_swizzled_packed mean time: 0.015785 ms, speedup: 1.64
####################################################################################################
n: 512, m: 2048
torch                          mean time: 0.044284 ms
transpose_coalesced_read       mean time: 0.017416 ms, speedup: 2.54
transpose_coalesced_write      mean time: 0.021171 ms, speedup: 2.09
transpose_smem                 mean time: 0.027755 ms, speedup: 1.60
transpose_smem_bcf             mean time: 0.013953 ms, speedup: 3.17
transpose_smem_packed_bcf      mean time: 0.010289 ms, speedup: 4.30
transpose_smem_swizzled_packed mean time: 0.012161 ms, speedup: 3.64
####################################################################################################
n: 512, m: 4096
torch                          mean time: 0.094803 ms
transpose_coalesced_read       mean time: 0.036812 ms, speedup: 2.58
transpose_coalesced_write      mean time: 0.026813 ms, speedup: 3.54
transpose_smem                 mean time: 0.043888 ms, speedup: 2.16
transpose_smem_bcf             mean time: 0.022350 ms, speedup: 4.24
transpose_smem_packed_bcf      mean time: 0.014386 ms, speedup: 6.59
transpose_smem_swizzled_packed mean time: 0.014151 ms, speedup: 6.70
####################################################################################################
n: 512, m: 8192
torch                          mean time: 0.210648 ms
transpose_coalesced_read       mean time: 0.077352 ms, speedup: 2.72
transpose_coalesced_write      mean time: 0.064794 ms, speedup: 3.25
transpose_smem                 mean time: 0.094334 ms, speedup: 2.23
transpose_smem_bcf             mean time: 0.027147 ms, speedup: 7.76
transpose_smem_packed_bcf      mean time: 0.027989 ms, speedup: 7.53
transpose_smem_swizzled_packed mean time: 0.034089 ms, speedup: 6.18
####################################################################################################
n: 1024, m: 256
torch                          mean time: 0.031145 ms
transpose_coalesced_read       mean time: 0.015020 ms, speedup: 2.07
transpose_coalesced_write      mean time: 0.013510 ms, speedup: 2.31
transpose_smem                 mean time: 0.018345 ms, speedup: 1.70
transpose_smem_bcf             mean time: 0.007449 ms, speedup: 4.18
transpose_smem_packed_bcf      mean time: 0.014985 ms, speedup: 2.08
transpose_smem_swizzled_packed mean time: 0.016323 ms, speedup: 1.91
####################################################################################################
n: 1024, m: 512
torch                          mean time: 0.033004 ms
transpose_coalesced_read       mean time: 0.012282 ms, speedup: 2.69
transpose_coalesced_write      mean time: 0.009849 ms, speedup: 3.35
transpose_smem                 mean time: 0.013117 ms, speedup: 2.52
transpose_smem_bcf             mean time: 0.010003 ms, speedup: 3.30
transpose_smem_packed_bcf      mean time: 0.008798 ms, speedup: 3.75
transpose_smem_swizzled_packed mean time: 0.017373 ms, speedup: 1.90
####################################################################################################
n: 1024, m: 1024
torch                          mean time: 0.049843 ms
transpose_coalesced_read       mean time: 0.015823 ms, speedup: 3.15
transpose_coalesced_write      mean time: 0.016530 ms, speedup: 3.02
transpose_smem                 mean time: 0.030072 ms, speedup: 1.66
transpose_smem_bcf             mean time: 0.011004 ms, speedup: 4.53
transpose_smem_packed_bcf      mean time: 0.016048 ms, speedup: 3.11
transpose_smem_swizzled_packed mean time: 0.013602 ms, speedup: 3.66
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.088150 ms
transpose_coalesced_read       mean time: 0.028716 ms, speedup: 3.07
transpose_coalesced_write      mean time: 0.027246 ms, speedup: 3.24
transpose_smem                 mean time: 0.051087 ms, speedup: 1.73
transpose_smem_bcf             mean time: 0.018593 ms, speedup: 4.74
transpose_smem_packed_bcf      mean time: 0.013986 ms, speedup: 6.30
transpose_smem_swizzled_packed mean time: 0.014857 ms, speedup: 5.93
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.223741 ms
transpose_coalesced_read       mean time: 0.072797 ms, speedup: 3.07
transpose_coalesced_write      mean time: 0.071779 ms, speedup: 3.12
transpose_smem                 mean time: 0.101503 ms, speedup: 2.20
transpose_smem_bcf             mean time: 0.027756 ms, speedup: 8.06
transpose_smem_packed_bcf      mean time: 0.028193 ms, speedup: 7.94
transpose_smem_swizzled_packed mean time: 0.040087 ms, speedup: 5.58
####################################################################################################
n: 1024, m: 8192
torch                          mean time: 0.525525 ms
transpose_coalesced_read       mean time: 0.250677 ms, speedup: 2.10
transpose_coalesced_write      mean time: 0.252549 ms, speedup: 2.08
transpose_smem                 mean time: 0.221449 ms, speedup: 2.37
transpose_smem_bcf             mean time: 0.226624 ms, speedup: 2.32
transpose_smem_packed_bcf      mean time: 0.220865 ms, speedup: 2.38
transpose_smem_swizzled_packed mean time: 0.228077 ms, speedup: 2.30
####################################################################################################
n: 2048, m: 256
torch                          mean time: 0.030520 ms
transpose_coalesced_read       mean time: 0.013874 ms, speedup: 2.20
transpose_coalesced_write      mean time: 0.012076 ms, speedup: 2.53
transpose_smem                 mean time: 0.020979 ms, speedup: 1.45
transpose_smem_bcf             mean time: 0.009684 ms, speedup: 3.15
transpose_smem_packed_bcf      mean time: 0.010636 ms, speedup: 2.87
transpose_smem_swizzled_packed mean time: 0.021859 ms, speedup: 1.40
####################################################################################################
n: 2048, m: 512
torch                          mean time: 0.049364 ms
transpose_coalesced_read       mean time: 0.015388 ms, speedup: 3.21
transpose_coalesced_write      mean time: 0.017926 ms, speedup: 2.75
transpose_smem                 mean time: 0.029574 ms, speedup: 1.67
transpose_smem_bcf             mean time: 0.011111 ms, speedup: 4.44
transpose_smem_packed_bcf      mean time: 0.011120 ms, speedup: 4.44
transpose_smem_swizzled_packed mean time: 0.008897 ms, speedup: 5.55
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.080915 ms
transpose_coalesced_read       mean time: 0.035458 ms, speedup: 2.28
transpose_coalesced_write      mean time: 0.026644 ms, speedup: 3.04
transpose_smem                 mean time: 0.042728 ms, speedup: 1.89
transpose_smem_bcf             mean time: 0.014556 ms, speedup: 5.56
transpose_smem_packed_bcf      mean time: 0.014873 ms, speedup: 5.44
transpose_smem_swizzled_packed mean time: 0.014168 ms, speedup: 5.71
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.241208 ms
transpose_coalesced_read       mean time: 0.072029 ms, speedup: 3.35
transpose_coalesced_write      mean time: 0.060311 ms, speedup: 4.00
transpose_smem                 mean time: 0.094270 ms, speedup: 2.56
transpose_smem_bcf             mean time: 0.036416 ms, speedup: 6.62
transpose_smem_packed_bcf      mean time: 0.027087 ms, speedup: 8.90
transpose_smem_swizzled_packed mean time: 0.029249 ms, speedup: 8.25
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.538468 ms
transpose_coalesced_read       mean time: 0.250143 ms, speedup: 2.15
transpose_coalesced_write      mean time: 0.259906 ms, speedup: 2.07
transpose_smem                 mean time: 0.220278 ms, speedup: 2.44
transpose_smem_bcf             mean time: 0.220552 ms, speedup: 2.44
transpose_smem_packed_bcf      mean time: 0.220418 ms, speedup: 2.44
transpose_smem_swizzled_packed mean time: 0.222534 ms, speedup: 2.42
####################################################################################################
n: 2048, m: 8192
torch                          mean time: 1.064801 ms
transpose_coalesced_read       mean time: 0.478184 ms, speedup: 2.23
transpose_coalesced_write      mean time: 0.471281 ms, speedup: 2.26
transpose_smem                 mean time: 0.438343 ms, speedup: 2.43
transpose_smem_bcf             mean time: 0.442842 ms, speedup: 2.40
transpose_smem_packed_bcf      mean time: 0.445173 ms, speedup: 2.39
transpose_smem_swizzled_packed mean time: 0.443150 ms, speedup: 2.40
####################################################################################################
n: 4096, m: 256
torch                          mean time: 0.043081 ms
transpose_coalesced_read       mean time: 0.018962 ms, speedup: 2.27
transpose_coalesced_write      mean time: 0.016445 ms, speedup: 2.62
transpose_smem                 mean time: 0.023285 ms, speedup: 1.85
transpose_smem_bcf             mean time: 0.011133 ms, speedup: 3.87
transpose_smem_packed_bcf      mean time: 0.017996 ms, speedup: 2.39
transpose_smem_swizzled_packed mean time: 0.011926 ms, speedup: 3.61
####################################################################################################
n: 4096, m: 512
torch                          mean time: 0.087197 ms
transpose_coalesced_read       mean time: 0.029422 ms, speedup: 2.96
transpose_coalesced_write      mean time: 0.027401 ms, speedup: 3.18
transpose_smem                 mean time: 0.048083 ms, speedup: 1.81
transpose_smem_bcf             mean time: 0.014509 ms, speedup: 6.01
transpose_smem_packed_bcf      mean time: 0.015595 ms, speedup: 5.59
transpose_smem_swizzled_packed mean time: 0.015189 ms, speedup: 5.74
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.240903 ms
transpose_coalesced_read       mean time: 0.063097 ms, speedup: 3.82
transpose_coalesced_write      mean time: 0.057016 ms, speedup: 4.23
transpose_smem                 mean time: 0.093666 ms, speedup: 2.57
transpose_smem_bcf             mean time: 0.026635 ms, speedup: 9.04
transpose_smem_packed_bcf      mean time: 0.027996 ms, speedup: 8.60
transpose_smem_swizzled_packed mean time: 0.036712 ms, speedup: 6.56
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.543797 ms
transpose_coalesced_read       mean time: 0.244612 ms, speedup: 2.22
transpose_coalesced_write      mean time: 0.246672 ms, speedup: 2.20
transpose_smem                 mean time: 0.218376 ms, speedup: 2.49
transpose_smem_bcf             mean time: 0.220551 ms, speedup: 2.47
transpose_smem_packed_bcf      mean time: 0.232066 ms, speedup: 2.34
transpose_smem_swizzled_packed mean time: 0.275900 ms, speedup: 1.97
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 1.089675 ms
transpose_coalesced_read       mean time: 0.464379 ms, speedup: 2.35
transpose_coalesced_write      mean time: 0.480570 ms, speedup: 2.27
transpose_smem                 mean time: 0.423648 ms, speedup: 2.57
transpose_smem_bcf             mean time: 0.426381 ms, speedup: 2.56
transpose_smem_packed_bcf      mean time: 0.426405 ms, speedup: 2.56
transpose_smem_swizzled_packed mean time: 0.413314 ms, speedup: 2.64
####################################################################################################
n: 4096, m: 8192
torch                          mean time: 2.122498 ms
transpose_coalesced_read       mean time: 0.999802 ms, speedup: 2.12
transpose_coalesced_write      mean time: 0.937398 ms, speedup: 2.26
transpose_smem                 mean time: 0.849731 ms, speedup: 2.50
transpose_smem_bcf             mean time: 0.870054 ms, speedup: 2.44
transpose_smem_packed_bcf      mean time: 0.855833 ms, speedup: 2.48
transpose_smem_swizzled_packed mean time: 0.860991 ms, speedup: 2.47
####################################################################################################
n: 8192, m: 256
torch                          mean time: 0.090407 ms
transpose_coalesced_read       mean time: 0.028363 ms, speedup: 3.19
transpose_coalesced_write      mean time: 0.027353 ms, speedup: 3.31
transpose_smem                 mean time: 0.047054 ms, speedup: 1.92
transpose_smem_bcf             mean time: 0.014593 ms, speedup: 6.20
transpose_smem_packed_bcf      mean time: 0.015198 ms, speedup: 5.95
transpose_smem_swizzled_packed mean time: 0.018953 ms, speedup: 4.77
####################################################################################################
n: 8192, m: 512
torch                          mean time: 0.236846 ms
transpose_coalesced_read       mean time: 0.057367 ms, speedup: 4.13
transpose_coalesced_write      mean time: 0.059867 ms, speedup: 3.96
transpose_smem                 mean time: 0.088823 ms, speedup: 2.67
transpose_smem_bcf             mean time: 0.025271 ms, speedup: 9.37
transpose_smem_packed_bcf      mean time: 0.026056 ms, speedup: 9.09
transpose_smem_swizzled_packed mean time: 0.034299 ms, speedup: 6.91
####################################################################################################
n: 8192, m: 1024
torch                          mean time: 0.534517 ms
transpose_coalesced_read       mean time: 0.225614 ms, speedup: 2.37
transpose_coalesced_write      mean time: 0.243479 ms, speedup: 2.20
transpose_smem                 mean time: 0.212052 ms, speedup: 2.52
transpose_smem_bcf             mean time: 0.211712 ms, speedup: 2.52
transpose_smem_packed_bcf      mean time: 0.216930 ms, speedup: 2.46
transpose_smem_swizzled_packed mean time: 0.216879 ms, speedup: 2.46
####################################################################################################
n: 8192, m: 2048
torch                          mean time: 1.088668 ms
transpose_coalesced_read       mean time: 0.461737 ms, speedup: 2.36
transpose_coalesced_write      mean time: 0.474453 ms, speedup: 2.29
transpose_smem                 mean time: 0.422272 ms, speedup: 2.58
transpose_smem_bcf             mean time: 0.410244 ms, speedup: 2.65
transpose_smem_packed_bcf      mean time: 0.422504 ms, speedup: 2.58
transpose_smem_swizzled_packed mean time: 0.409407 ms, speedup: 2.66
####################################################################################################
n: 8192, m: 4096
torch                          mean time: 2.112788 ms
transpose_coalesced_read       mean time: 0.965073 ms, speedup: 2.19
transpose_coalesced_write      mean time: 0.988541 ms, speedup: 2.14
transpose_smem                 mean time: 0.824117 ms, speedup: 2.56
transpose_smem_bcf             mean time: 0.833272 ms, speedup: 2.54
transpose_smem_packed_bcf      mean time: 0.825299 ms, speedup: 2.56
transpose_smem_swizzled_packed mean time: 0.833656 ms, speedup: 2.53
####################################################################################################
n: 8192, m: 8192
torch                          mean time: 4.871119 ms
transpose_coalesced_read       mean time: 2.063641 ms, speedup: 2.36
transpose_coalesced_write      mean time: 1.883395 ms, speedup: 2.59
transpose_smem                 mean time: 1.708258 ms, speedup: 2.85
transpose_smem_bcf             mean time: 1.690630 ms, speedup: 2.88
transpose_smem_packed_bcf      mean time: 1.698521 ms, speedup: 2.87
transpose_smem_swizzled_packed mean time: 1.698522 ms, speedup: 2.87
```
