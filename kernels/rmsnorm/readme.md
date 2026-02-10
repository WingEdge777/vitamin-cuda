# rmsnorm

## 说明

rmsnorm kernel

- [x] naive torch rmsnorm
- [x] rmsnorm fp32/fp16 版
- [x] rmsnorm fp32x4 版 (fp32向量化)
- [x] rmsnorm fp16x8 版 (fp16向量化, packed r/w)
- [x] pytorch op bindings && diff check

## 测试

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### 输出

```bash
####################################################################################################
n: 256, m: 2048
torch                          mean time: 0.078499 ms
rmsnorm                        mean time: 0.014753 ms
rmsnorm_fp32x4                 mean time: 0.014874 ms
torch                          mean time: 0.105773 ms
rmsnorm_fp16                   mean time: 0.014875 ms
rmsnorm_fp16x8_packed          mean time: 0.019436 ms
####################################################################################################
n: 256, m: 4096
torch                          mean time: 0.111625 ms
rmsnorm                        mean time: 0.023459 ms
rmsnorm_fp32x4                 mean time: 0.021768 ms
torch                          mean time: 0.114969 ms
rmsnorm_fp16                   mean time: 0.017503 ms
rmsnorm_fp16x8_packed          mean time: 0.023904 ms
####################################################################################################
n: 256, m: 8192
torch                          mean time: 0.154215 ms
rmsnorm                        mean time: 0.037323 ms
rmsnorm_fp32x4                 mean time: 0.037992 ms
torch                          mean time: 0.118463 ms
rmsnorm_fp16                   mean time: 0.033754 ms
rmsnorm_fp16x8_packed          mean time: 0.033779 ms
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.134233 ms
rmsnorm                        mean time: 0.047741 ms
rmsnorm_fp32x4                 mean time: 0.035438 ms
torch                          mean time: 0.124022 ms
rmsnorm_fp16                   mean time: 0.039586 ms
rmsnorm_fp16x8_packed          mean time: 0.023664 ms
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.334243 ms
rmsnorm                        mean time: 0.139253 ms
rmsnorm_fp32x4                 mean time: 0.138813 ms
torch                          mean time: 0.227061 ms
rmsnorm_fp16                   mean time: 0.057913 ms
rmsnorm_fp16x8_packed          mean time: 0.044938 ms
####################################################################################################
n: 1024, m: 8192
torch                          mean time: 0.908917 ms
rmsnorm                        mean time: 0.293798 ms
rmsnorm_fp32x4                 mean time: 0.298978 ms
torch                          mean time: 0.446633 ms
rmsnorm_fp16                   mean time: 0.135620 ms
rmsnorm_fp16x8_packed          mean time: 0.139449 ms
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.310171 ms
rmsnorm                        mean time: 0.139068 ms
rmsnorm_fp32x4                 mean time: 0.158098 ms
torch                          mean time: 0.184838 ms
rmsnorm_fp16                   mean time: 0.060903 ms
rmsnorm_fp16x8_packed          mean time: 0.033342 ms
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.961401 ms
rmsnorm                        mean time: 0.322991 ms
rmsnorm_fp32x4                 mean time: 0.277981 ms
torch                          mean time: 0.423590 ms
rmsnorm_fp16                   mean time: 0.148105 ms
rmsnorm_fp16x8_packed          mean time: 0.157540 ms
####################################################################################################
n: 2048, m: 8192
torch                          mean time: 1.943526 ms
rmsnorm                        mean time: 0.600290 ms
rmsnorm_fp32x4                 mean time: 0.570166 ms
torch                          mean time: 1.093987 ms
rmsnorm_fp16                   mean time: 0.293870 ms
rmsnorm_fp16x8_packed          mean time: 0.277978 ms
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.888998 ms
rmsnorm                        mean time: 0.277658 ms
rmsnorm_fp32x4                 mean time: 0.284801 ms
torch                          mean time: 0.354362 ms
rmsnorm_fp16                   mean time: 0.147305 ms
rmsnorm_fp16x8_packed          mean time: 0.139002 ms
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 1.861076 ms
rmsnorm                        mean time: 0.569079 ms
rmsnorm_fp32x4                 mean time: 0.540098 ms
torch                          mean time: 0.941507 ms
rmsnorm_fp16                   mean time: 0.282269 ms
rmsnorm_fp16x8_packed          mean time: 0.279492 ms
####################################################################################################
n: 4096, m: 8192
torch                          mean time: 4.169647 ms
rmsnorm                        mean time: 1.137849 ms
rmsnorm_fp32x4                 mean time: 1.066910 ms
torch                          mean time: 2.010596 ms
rmsnorm_fp16                   mean time: 0.559046 ms
rmsnorm_fp16x8_packed          mean time: 0.543035 ms
```
