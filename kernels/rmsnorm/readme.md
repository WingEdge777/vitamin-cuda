# rmsnorm

## 说明

rmsnorm kernel

- [x] naive torch rmsnorm
- [x] rmsnorm fp32/fp16 版
- [x] rmsnorm fp32x4 版 (fp32向量化)
- [x] rmsnorm_fp32x4_smem
- [x] rmsnorm fp16x8 版 (fp16向量化, packed r/w)
- [x] rmsnorm_fp16x8_smem 版 (fp16向量化, packed r/w)
- [x] pytorch op bindings && diff check

## 测试

L2 cache 比较大,smem优化权重读取没什么效果,瓶颈还是在input/output读写上

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### 输出

```bash
####################################################################################################
n: 256, m: 2048
torch                          mean time: 0.065773 ms
rmsnorm                        mean time: 0.012925 ms
rmsnorm_fp32x4                 mean time: 0.034686 ms
rmsnorm_fp32x4_smem            mean time: 0.026763 ms
torch                          mean time: 0.077993 ms
rmsnorm_fp16                   mean time: 0.009886 ms
rmsnorm_fp16x8_packed          mean time: 0.008307 ms
rmsnorm_fp16x8_packed_smem     mean time: 0.013596 ms
####################################################################################################
n: 256, m: 4096
torch                          mean time: 0.074318 ms
rmsnorm                        mean time: 0.014684 ms
rmsnorm_fp32x4                 mean time: 0.021912 ms
rmsnorm_fp32x4_smem            mean time: 0.045206 ms
torch                          mean time: 0.168281 ms
rmsnorm_fp16                   mean time: 0.030428 ms
rmsnorm_fp16x8_packed          mean time: 0.018289 ms
rmsnorm_fp16x8_packed_smem     mean time: 0.018621 ms
####################################################################################################
n: 256, m: 8192
torch                          mean time: 0.103797 ms
rmsnorm                        mean time: 0.026693 ms
rmsnorm_fp32x4                 mean time: 0.029177 ms
rmsnorm_fp32x4_smem            mean time: 0.053339 ms
torch                          mean time: 0.107911 ms
rmsnorm_fp16                   mean time: 0.027226 ms
rmsnorm_fp16x8_packed          mean time: 0.015751 ms
rmsnorm_fp16x8_packed_smem     mean time: 0.029455 ms
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.092873 ms
rmsnorm                        mean time: 0.024123 ms
rmsnorm_fp32x4                 mean time: 0.029974 ms
rmsnorm_fp32x4_smem            mean time: 0.036221 ms
torch                          mean time: 0.107078 ms
rmsnorm_fp16                   mean time: 0.020840 ms
rmsnorm_fp16x8_packed          mean time: 0.020286 ms
rmsnorm_fp16x8_packed_smem     mean time: 0.017987 ms
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.282271 ms
rmsnorm                        mean time: 0.122600 ms
rmsnorm_fp32x4                 mean time: 0.115992 ms
rmsnorm_fp32x4_smem            mean time: 0.118212 ms
torch                          mean time: 0.132647 ms
rmsnorm_fp16                   mean time: 0.037972 ms
rmsnorm_fp16x8_packed          mean time: 0.029384 ms
rmsnorm_fp16x8_packed_smem     mean time: 0.043331 ms
####################################################################################################
n: 1024, m: 8192
torch                          mean time: 0.693848 ms
rmsnorm                        mean time: 0.233323 ms
rmsnorm_fp32x4                 mean time: 0.213837 ms
rmsnorm_fp32x4_smem            mean time: 0.232781 ms
torch                          mean time: 0.363583 ms
rmsnorm_fp16                   mean time: 0.120641 ms
rmsnorm_fp16x8_packed          mean time: 0.107922 ms
rmsnorm_fp16x8_packed_smem     mean time: 0.103317 ms
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.240520 ms
rmsnorm                        mean time: 0.107789 ms
rmsnorm_fp32x4                 mean time: 0.132142 ms
rmsnorm_fp32x4_smem            mean time: 0.104028 ms
torch                          mean time: 0.131029 ms
rmsnorm_fp16                   mean time: 0.036482 ms
rmsnorm_fp16x8_packed          mean time: 0.041102 ms
rmsnorm_fp16x8_packed_smem     mean time: 0.037262 ms
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.809969 ms
rmsnorm                        mean time: 0.230314 ms
rmsnorm_fp32x4                 mean time: 0.236344 ms
rmsnorm_fp32x4_smem            mean time: 0.245532 ms
torch                          mean time: 0.341274 ms
rmsnorm_fp16                   mean time: 0.106764 ms
rmsnorm_fp16x8_packed          mean time: 0.125952 ms
rmsnorm_fp16x8_packed_smem     mean time: 0.124710 ms
####################################################################################################
n: 2048, m: 8192
torch                          mean time: 1.407594 ms
rmsnorm                        mean time: 0.448651 ms
rmsnorm_fp32x4                 mean time: 0.443729 ms
rmsnorm_fp32x4_smem            mean time: 0.534574 ms
torch                          mean time: 1.008208 ms
rmsnorm_fp16                   mean time: 0.237907 ms
rmsnorm_fp16x8_packed          mean time: 0.228068 ms
rmsnorm_fp16x8_packed_smem     mean time: 0.241439 ms
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.689149 ms
rmsnorm                        mean time: 0.234225 ms
rmsnorm_fp32x4                 mean time: 0.222221 ms
rmsnorm_fp32x4_smem            mean time: 0.238920 ms
torch                          mean time: 0.399793 ms
rmsnorm_fp16                   mean time: 0.121430 ms
rmsnorm_fp16x8_packed          mean time: 0.124461 ms
rmsnorm_fp16x8_packed_smem     mean time: 0.094941 ms
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 1.411146 ms
rmsnorm                        mean time: 0.473826 ms
rmsnorm_fp32x4                 mean time: 0.435659 ms
rmsnorm_fp32x4_smem            mean time: 0.446924 ms
torch                          mean time: 0.994632 ms
rmsnorm_fp16                   mean time: 0.296433 ms
rmsnorm_fp16x8_packed          mean time: 0.292735 ms
rmsnorm_fp16x8_packed_smem     mean time: 0.288811 ms
####################################################################################################
n: 4096, m: 8192
torch                          mean time: 3.143430 ms
rmsnorm                        mean time: 1.048659 ms
rmsnorm_fp32x4                 mean time: 0.918590 ms
rmsnorm_fp32x4_smem            mean time: 0.902087 ms
torch                          mean time: 1.963521 ms
rmsnorm_fp16                   mean time: 0.467633 ms
rmsnorm_fp16x8_packed          mean time: 0.456928 ms
rmsnorm_fp16x8_packed_smem     mean time: 0.472404 ms
```
