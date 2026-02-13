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
torch                          mean time: 0.017915 ms
transpose_coalesced_read       mean time: 0.008513 ms, speedup: 2.10
transpose_coalesced_write      mean time: 0.009208 ms, speedup: 1.95
transpose_smem                 mean time: 0.008831 ms, speedup: 2.03
transpose_smem_bcf             mean time: 0.006912 ms, speedup: 2.59
transpose_smem_packed_bcf      mean time: 0.007402 ms, speedup: 2.42
transpose_smem_swizzled_packed mean time: 0.009434 ms, speedup: 1.90
####################################################################################################
n: 256, m: 512
torch                          mean time: 0.044788 ms
transpose_coalesced_read       mean time: 0.008430 ms, speedup: 5.31
transpose_coalesced_write      mean time: 0.008373 ms, speedup: 5.35
transpose_smem                 mean time: 0.009561 ms, speedup: 4.68
transpose_smem_bcf             mean time: 0.007186 ms, speedup: 6.23
transpose_smem_packed_bcf      mean time: 0.007719 ms, speedup: 5.80
transpose_smem_swizzled_packed mean time: 0.013790 ms, speedup: 3.25
####################################################################################################
n: 256, m: 1024
torch                          mean time: 0.027871 ms
transpose_coalesced_read       mean time: 0.009415 ms, speedup: 2.96
transpose_coalesced_write      mean time: 0.010102 ms, speedup: 2.76
transpose_smem                 mean time: 0.011323 ms, speedup: 2.46
transpose_smem_bcf             mean time: 0.008588 ms, speedup: 3.25
transpose_smem_packed_bcf      mean time: 0.008558 ms, speedup: 3.26
transpose_smem_swizzled_packed mean time: 0.007379 ms, speedup: 3.78
####################################################################################################
n: 256, m: 2048
torch                          mean time: 0.030743 ms
transpose_coalesced_read       mean time: 0.012547 ms, speedup: 2.45
transpose_coalesced_write      mean time: 0.013480 ms, speedup: 2.28
transpose_smem                 mean time: 0.015261 ms, speedup: 2.01
transpose_smem_bcf             mean time: 0.010996 ms, speedup: 2.80
transpose_smem_packed_bcf      mean time: 0.012218 ms, speedup: 2.52
transpose_smem_swizzled_packed mean time: 0.009767 ms, speedup: 3.15
####################################################################################################
n: 256, m: 4096
torch                          mean time: 0.046995 ms
transpose_coalesced_read       mean time: 0.019267 ms, speedup: 2.44
transpose_coalesced_write      mean time: 0.016383 ms, speedup: 2.87
transpose_smem                 mean time: 0.025224 ms, speedup: 1.86
transpose_smem_bcf             mean time: 0.010255 ms, speedup: 4.58
transpose_smem_packed_bcf      mean time: 0.010378 ms, speedup: 4.53
transpose_smem_swizzled_packed mean time: 0.011300 ms, speedup: 4.16
####################################################################################################
n: 256, m: 8192
torch                          mean time: 0.091893 ms
transpose_coalesced_read       mean time: 0.031320 ms, speedup: 2.93
transpose_coalesced_write      mean time: 0.028541 ms, speedup: 3.22
transpose_smem                 mean time: 0.043726 ms, speedup: 2.10
transpose_smem_bcf             mean time: 0.016285 ms, speedup: 5.64
transpose_smem_packed_bcf      mean time: 0.016313 ms, speedup: 5.63
transpose_smem_swizzled_packed mean time: 0.015768 ms, speedup: 5.83
####################################################################################################
n: 512, m: 256
torch                          mean time: 0.021401 ms
transpose_coalesced_read       mean time: 0.008426 ms, speedup: 2.54
transpose_coalesced_write      mean time: 0.009870 ms, speedup: 2.17
transpose_smem                 mean time: 0.010725 ms, speedup: 2.00
transpose_smem_bcf             mean time: 0.009021 ms, speedup: 2.37
transpose_smem_packed_bcf      mean time: 0.008066 ms, speedup: 2.65
transpose_smem_swizzled_packed mean time: 0.007183 ms, speedup: 2.98
####################################################################################################
n: 512, m: 512
torch                          mean time: 0.021054 ms
transpose_coalesced_read       mean time: 0.011268 ms, speedup: 1.87
transpose_coalesced_write      mean time: 0.010086 ms, speedup: 2.09
transpose_smem                 mean time: 0.011633 ms, speedup: 1.81
transpose_smem_bcf             mean time: 0.008860 ms, speedup: 2.38
transpose_smem_packed_bcf      mean time: 0.009331 ms, speedup: 2.26
transpose_smem_swizzled_packed mean time: 0.009475 ms, speedup: 2.22
####################################################################################################
n: 512, m: 1024
torch                          mean time: 0.030537 ms
transpose_coalesced_read       mean time: 0.013481 ms, speedup: 2.27
transpose_coalesced_write      mean time: 0.014104 ms, speedup: 2.17
transpose_smem                 mean time: 0.013712 ms, speedup: 2.23
transpose_smem_bcf             mean time: 0.008640 ms, speedup: 3.53
transpose_smem_packed_bcf      mean time: 0.009009 ms, speedup: 3.39
transpose_smem_swizzled_packed mean time: 0.011052 ms, speedup: 2.76
####################################################################################################
n: 512, m: 2048
torch                          mean time: 0.042589 ms
transpose_coalesced_read       mean time: 0.015810 ms, speedup: 2.69
transpose_coalesced_write      mean time: 0.018034 ms, speedup: 2.36
transpose_smem                 mean time: 0.024942 ms, speedup: 1.71
transpose_smem_bcf             mean time: 0.011988 ms, speedup: 3.55
transpose_smem_packed_bcf      mean time: 0.011572 ms, speedup: 3.68
transpose_smem_swizzled_packed mean time: 0.010627 ms, speedup: 4.01
####################################################################################################
n: 512, m: 4096
torch                          mean time: 0.079577 ms
transpose_coalesced_read       mean time: 0.031022 ms, speedup: 2.57
transpose_coalesced_write      mean time: 0.028634 ms, speedup: 2.78
transpose_smem                 mean time: 0.043995 ms, speedup: 1.81
transpose_smem_bcf             mean time: 0.016194 ms, speedup: 4.91
transpose_smem_packed_bcf      mean time: 0.014804 ms, speedup: 5.38
transpose_smem_swizzled_packed mean time: 0.015450 ms, speedup: 5.15
####################################################################################################
n: 512, m: 8192
torch                          mean time: 0.199207 ms
transpose_coalesced_read       mean time: 0.074660 ms, speedup: 2.67
transpose_coalesced_write      mean time: 0.062594 ms, speedup: 3.18
transpose_smem                 mean time: 0.094927 ms, speedup: 2.10
transpose_smem_bcf             mean time: 0.028714 ms, speedup: 6.94
transpose_smem_packed_bcf      mean time: 0.029392 ms, speedup: 6.78
transpose_smem_swizzled_packed mean time: 0.031405 ms, speedup: 6.34
####################################################################################################
n: 1024, m: 256
torch                          mean time: 0.029700 ms
transpose_coalesced_read       mean time: 0.011967 ms, speedup: 2.48
transpose_coalesced_write      mean time: 0.010772 ms, speedup: 2.76
transpose_smem                 mean time: 0.011352 ms, speedup: 2.62
transpose_smem_bcf             mean time: 0.008610 ms, speedup: 3.45
transpose_smem_packed_bcf      mean time: 0.009766 ms, speedup: 3.04
transpose_smem_swizzled_packed mean time: 0.007351 ms, speedup: 4.04
####################################################################################################
n: 1024, m: 512
torch                          mean time: 0.026390 ms
transpose_coalesced_read       mean time: 0.022723 ms, speedup: 1.16
transpose_coalesced_write      mean time: 0.012622 ms, speedup: 2.09
transpose_smem                 mean time: 0.015626 ms, speedup: 1.69
transpose_smem_bcf             mean time: 0.012404 ms, speedup: 2.13
transpose_smem_packed_bcf      mean time: 0.009783 ms, speedup: 2.70
transpose_smem_swizzled_packed mean time: 0.009160 ms, speedup: 2.88
####################################################################################################
n: 1024, m: 1024
torch                          mean time: 0.042705 ms
transpose_coalesced_read       mean time: 0.015138 ms, speedup: 2.82
transpose_coalesced_write      mean time: 0.014399 ms, speedup: 2.97
transpose_smem                 mean time: 0.023274 ms, speedup: 1.83
transpose_smem_bcf             mean time: 0.009114 ms, speedup: 4.69
transpose_smem_packed_bcf      mean time: 0.009491 ms, speedup: 4.50
transpose_smem_swizzled_packed mean time: 0.010085 ms, speedup: 4.23
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.086052 ms
transpose_coalesced_read       mean time: 0.030865 ms, speedup: 2.79
transpose_coalesced_write      mean time: 0.028242 ms, speedup: 3.05
transpose_smem                 mean time: 0.046137 ms, speedup: 1.87
transpose_smem_bcf             mean time: 0.015299 ms, speedup: 5.62
transpose_smem_packed_bcf      mean time: 0.016494 ms, speedup: 5.22
transpose_smem_swizzled_packed mean time: 0.014538 ms, speedup: 5.92
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.211869 ms
transpose_coalesced_read       mean time: 0.074561 ms, speedup: 2.84
transpose_coalesced_write      mean time: 0.063498 ms, speedup: 3.34
transpose_smem                 mean time: 0.095665 ms, speedup: 2.21
transpose_smem_bcf             mean time: 0.031635 ms, speedup: 6.70
transpose_smem_packed_bcf      mean time: 0.031188 ms, speedup: 6.79
transpose_smem_swizzled_packed mean time: 0.031889 ms, speedup: 6.64
####################################################################################################
n: 1024, m: 8192
torch                          mean time: 0.530252 ms
transpose_coalesced_read       mean time: 0.255766 ms, speedup: 2.07
transpose_coalesced_write      mean time: 0.256969 ms, speedup: 2.06
transpose_smem                 mean time: 0.230783 ms, speedup: 2.30
transpose_smem_bcf             mean time: 0.225492 ms, speedup: 2.35
transpose_smem_packed_bcf      mean time: 0.218522 ms, speedup: 2.43
transpose_smem_swizzled_packed mean time: 0.217077 ms, speedup: 2.44
####################################################################################################
n: 2048, m: 256
torch                          mean time: 0.032889 ms
transpose_coalesced_read       mean time: 0.011574 ms, speedup: 2.84
transpose_coalesced_write      mean time: 0.011745 ms, speedup: 2.80
transpose_smem                 mean time: 0.014468 ms, speedup: 2.27
transpose_smem_bcf             mean time: 0.011269 ms, speedup: 2.92
transpose_smem_packed_bcf      mean time: 0.008615 ms, speedup: 3.82
transpose_smem_swizzled_packed mean time: 0.008491 ms, speedup: 3.87
####################################################################################################
n: 2048, m: 512
torch                          mean time: 0.044881 ms
transpose_coalesced_read       mean time: 0.016145 ms, speedup: 2.78
transpose_coalesced_write      mean time: 0.016990 ms, speedup: 2.64
transpose_smem                 mean time: 0.023376 ms, speedup: 1.92
transpose_smem_bcf             mean time: 0.011314 ms, speedup: 3.97
transpose_smem_packed_bcf      mean time: 0.011125 ms, speedup: 4.03
transpose_smem_swizzled_packed mean time: 0.013084 ms, speedup: 3.43
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.082577 ms
transpose_coalesced_read       mean time: 0.030564 ms, speedup: 2.70
transpose_coalesced_write      mean time: 0.030163 ms, speedup: 2.74
transpose_smem                 mean time: 0.045783 ms, speedup: 1.80
transpose_smem_bcf             mean time: 0.015719 ms, speedup: 5.25
transpose_smem_packed_bcf      mean time: 0.015053 ms, speedup: 5.49
transpose_smem_swizzled_packed mean time: 0.015570 ms, speedup: 5.30
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.236453 ms
transpose_coalesced_read       mean time: 0.082497 ms, speedup: 2.87
transpose_coalesced_write      mean time: 0.068491 ms, speedup: 3.45
transpose_smem                 mean time: 0.101178 ms, speedup: 2.34
transpose_smem_bcf             mean time: 0.030913 ms, speedup: 7.65
transpose_smem_packed_bcf      mean time: 0.028839 ms, speedup: 8.20
transpose_smem_swizzled_packed mean time: 0.033192 ms, speedup: 7.12
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.550178 ms
transpose_coalesced_read       mean time: 0.247446 ms, speedup: 2.22
transpose_coalesced_write      mean time: 0.247375 ms, speedup: 2.22
transpose_smem                 mean time: 0.209915 ms, speedup: 2.62
transpose_smem_bcf             mean time: 0.205901 ms, speedup: 2.67
transpose_smem_packed_bcf      mean time: 0.217671 ms, speedup: 2.53
transpose_smem_swizzled_packed mean time: 0.225289 ms, speedup: 2.44
####################################################################################################
n: 2048, m: 8192
torch                          mean time: 1.112649 ms
transpose_coalesced_read       mean time: 0.497315 ms, speedup: 2.24
transpose_coalesced_write      mean time: 0.477799 ms, speedup: 2.33
transpose_smem                 mean time: 0.480561 ms, speedup: 2.32
transpose_smem_bcf             mean time: 0.424384 ms, speedup: 2.62
transpose_smem_packed_bcf      mean time: 0.425600 ms, speedup: 2.61
transpose_smem_swizzled_packed mean time: 0.425049 ms, speedup: 2.62
####################################################################################################
n: 4096, m: 256
torch                          mean time: 0.043327 ms
transpose_coalesced_read       mean time: 0.016544 ms, speedup: 2.62
transpose_coalesced_write      mean time: 0.016335 ms, speedup: 2.65
transpose_smem                 mean time: 0.022669 ms, speedup: 1.91
transpose_smem_bcf             mean time: 0.011114 ms, speedup: 3.90
transpose_smem_packed_bcf      mean time: 0.010672 ms, speedup: 4.06
transpose_smem_swizzled_packed mean time: 0.011193 ms, speedup: 3.87
####################################################################################################
n: 4096, m: 512
torch                          mean time: 0.084745 ms
transpose_coalesced_read       mean time: 0.031084 ms, speedup: 2.73
transpose_coalesced_write      mean time: 0.027665 ms, speedup: 3.06
transpose_smem                 mean time: 0.043354 ms, speedup: 1.95
transpose_smem_bcf             mean time: 0.015166 ms, speedup: 5.59
transpose_smem_packed_bcf      mean time: 0.016293 ms, speedup: 5.20
transpose_smem_swizzled_packed mean time: 0.015365 ms, speedup: 5.52
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.231038 ms
transpose_coalesced_read       mean time: 0.068236 ms, speedup: 3.39
transpose_coalesced_write      mean time: 0.060593 ms, speedup: 3.81
transpose_smem                 mean time: 0.093159 ms, speedup: 2.48
transpose_smem_bcf             mean time: 0.034649 ms, speedup: 6.67
transpose_smem_packed_bcf      mean time: 0.027774 ms, speedup: 8.32
transpose_smem_swizzled_packed mean time: 0.031783 ms, speedup: 7.27
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.558869 ms
transpose_coalesced_read       mean time: 0.252101 ms, speedup: 2.22
transpose_coalesced_write      mean time: 0.239827 ms, speedup: 2.33
transpose_smem                 mean time: 0.227927 ms, speedup: 2.45
transpose_smem_bcf             mean time: 0.205047 ms, speedup: 2.73
transpose_smem_packed_bcf      mean time: 0.204430 ms, speedup: 2.73
transpose_smem_swizzled_packed mean time: 0.204856 ms, speedup: 2.73
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 1.152674 ms
transpose_coalesced_read       mean time: 0.477185 ms, speedup: 2.42
transpose_coalesced_write      mean time: 0.500227 ms, speedup: 2.30
transpose_smem                 mean time: 0.418650 ms, speedup: 2.75
transpose_smem_bcf             mean time: 0.538088 ms, speedup: 2.14
transpose_smem_packed_bcf      mean time: 0.491356 ms, speedup: 2.35
transpose_smem_swizzled_packed mean time: 0.450949 ms, speedup: 2.56
####################################################################################################
n: 4096, m: 8192
torch                          mean time: 2.860153 ms
transpose_coalesced_read       mean time: 1.141564 ms, speedup: 2.51
transpose_coalesced_write      mean time: 0.942130 ms, speedup: 3.04
transpose_smem                 mean time: 0.885916 ms, speedup: 3.23
transpose_smem_bcf             mean time: 0.921508 ms, speedup: 3.10
transpose_smem_packed_bcf      mean time: 0.938095 ms, speedup: 3.05
transpose_smem_swizzled_packed mean time: 0.969484 ms, speedup: 2.95
####################################################################################################
n: 8192, m: 256
torch                          mean time: 0.095996 ms
transpose_coalesced_read       mean time: 0.034458 ms, speedup: 2.79
transpose_coalesced_write      mean time: 0.034082 ms, speedup: 2.82
transpose_smem                 mean time: 0.047705 ms, speedup: 2.01
transpose_smem_bcf             mean time: 0.025186 ms, speedup: 3.81
transpose_smem_packed_bcf      mean time: 0.027207 ms, speedup: 3.53
transpose_smem_swizzled_packed mean time: 0.023705 ms, speedup: 4.05
####################################################################################################
n: 8192, m: 512
torch                          mean time: 0.257472 ms
transpose_coalesced_read       mean time: 0.082064 ms, speedup: 3.14
transpose_coalesced_write      mean time: 0.072457 ms, speedup: 3.55
transpose_smem                 mean time: 0.107095 ms, speedup: 2.40
transpose_smem_bcf             mean time: 0.032369 ms, speedup: 7.95
transpose_smem_packed_bcf      mean time: 0.031282 ms, speedup: 8.23
transpose_smem_swizzled_packed mean time: 0.032217 ms, speedup: 7.99
####################################################################################################
n: 8192, m: 1024
torch                          mean time: 0.584660 ms
transpose_coalesced_read       mean time: 0.247958 ms, speedup: 2.36
transpose_coalesced_write      mean time: 0.239332 ms, speedup: 2.44
transpose_smem                 mean time: 0.221266 ms, speedup: 2.64
transpose_smem_bcf             mean time: 0.215293 ms, speedup: 2.72
transpose_smem_packed_bcf      mean time: 0.219078 ms, speedup: 2.67
transpose_smem_swizzled_packed mean time: 0.212451 ms, speedup: 2.75
####################################################################################################
n: 8192, m: 2048
torch                          mean time: 1.132851 ms
transpose_coalesced_read       mean time: 0.519010 ms, speedup: 2.18
transpose_coalesced_write      mean time: 0.506867 ms, speedup: 2.24
transpose_smem                 mean time: 0.480647 ms, speedup: 2.36
transpose_smem_bcf             mean time: 0.445889 ms, speedup: 2.54
transpose_smem_packed_bcf      mean time: 0.418792 ms, speedup: 2.71
transpose_smem_swizzled_packed mean time: 0.414753 ms, speedup: 2.73
####################################################################################################
n: 8192, m: 4096
torch                          mean time: 2.334238 ms
transpose_coalesced_read       mean time: 1.149069 ms, speedup: 2.03
transpose_coalesced_write      mean time: 1.024030 ms, speedup: 2.28
transpose_smem                 mean time: 0.893990 ms, speedup: 2.61
transpose_smem_bcf             mean time: 0.967593 ms, speedup: 2.41
transpose_smem_packed_bcf      mean time: 0.959794 ms, speedup: 2.43
transpose_smem_swizzled_packed mean time: 0.865132 ms, speedup: 2.70
####################################################################################################
n: 8192, m: 8192
torch                          mean time: 4.671009 ms
transpose_coalesced_read       mean time: 1.932761 ms, speedup: 2.42
transpose_coalesced_write      mean time: 1.845628 ms, speedup: 2.53
transpose_smem                 mean time: 1.706025 ms, speedup: 2.74
transpose_smem_bcf             mean time: 1.709746 ms, speedup: 2.73
transpose_smem_packed_bcf      mean time: 1.709723 ms, speedup: 2.73
transpose_smem_swizzled_packed mean time: 1.709305 ms, speedup: 2.73
```
