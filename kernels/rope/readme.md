# Rope

## 说明

rope kernel

- [x] pytorch naive rope
- [x] pytorch rope with cos/sin table
- [x] rope fp32 版 (比pytorch naive 实现快一个数量级)
- [x] rope fp32x4 版 (fp32向量化，快几十倍)
- [x] pytorch op bindings && diff check

## 测试

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### 输出

```bash
####################################################################################################
bs: 64, n: 512, m: 128
torch                          mean time: 0.630956 ms
torch.rope_with_sin_cos_cache  mean time: 0.451518 ms
rope                           mean time: 0.043930 ms
rope_fp32x4                    mean time: 0.026483 ms
####################################################################################################
bs: 64, n: 512, m: 256
torch                          mean time: 1.434311 ms
torch.rope_with_sin_cos_cache  mean time: 1.110631 ms
rope                           mean time: 0.215355 ms
rope_fp32x4                    mean time: 0.217933 ms
####################################################################################################
bs: 64, n: 1024, m: 128
torch                          mean time: 1.310259 ms
torch.rope_with_sin_cos_cache  mean time: 1.115163 ms
rope                           mean time: 0.194758 ms
rope_fp32x4                    mean time: 0.226546 ms
####################################################################################################
bs: 64, n: 1024, m: 256
torch                          mean time: 2.507560 ms
torch.rope_with_sin_cos_cache  mean time: 2.248296 ms
rope                           mean time: 0.420052 ms
rope_fp32x4                    mean time: 0.454816 ms
####################################################################################################
bs: 64, n: 2048, m: 128
torch                          mean time: 2.491320 ms
torch.rope_with_sin_cos_cache  mean time: 2.160297 ms
rope                           mean time: 0.426845 ms
rope_fp32x4                    mean time: 0.426953 ms
####################################################################################################
bs: 64, n: 2048, m: 256
torch                          mean time: 4.840254 ms
torch.rope_with_sin_cos_cache  mean time: 4.394541 ms
rope                           mean time: 0.868295 ms
rope_fp32x4                    mean time: 0.863909 ms
####################################################################################################
bs: 64, n: 4096, m: 128
torch                          mean time: 4.638200 ms
torch.rope_with_sin_cos_cache  mean time: 4.181268 ms
rope                           mean time: 0.823777 ms
rope_fp32x4                    mean time: 0.824729 ms
####################################################################################################
bs: 64, n: 4096, m: 256
torch                          mean time: 9.631756 ms
torch.rope_with_sin_cos_cache  mean time: 8.863726 ms
rope                           mean time: 1.660475 ms
rope_fp32x4                    mean time: 1.600244 ms
####################################################################################################
bs: 64, n: 8192, m: 128
torch                          mean time: 9.672222 ms
torch.rope_with_sin_cos_cache  mean time: 9.082619 ms
rope                           mean time: 1.736071 ms
rope_fp32x4                    mean time: 1.641156 ms
####################################################################################################
bs: 64, n: 8192, m: 256
torch                          mean time: 21.060976 ms
torch.rope_with_sin_cos_cache  mean time: 20.101883 ms
rope                           mean time: 3.201299 ms
rope_fp32x4                    mean time: 3.107204 ms
####################################################################################################
bs: 128, n: 512, m: 128
torch                          mean time: 1.223802 ms
torch.rope_with_sin_cos_cache  mean time: 1.020284 ms
rope                           mean time: 0.194133 ms
rope_fp32x4                    mean time: 0.218953 ms
####################################################################################################
bs: 128, n: 512, m: 256
torch                          mean time: 2.318807 ms
torch.rope_with_sin_cos_cache  mean time: 2.065181 ms
rope                           mean time: 0.408201 ms
rope_fp32x4                    mean time: 0.387216 ms
####################################################################################################
bs: 128, n: 1024, m: 128
torch                          mean time: 2.312312 ms
torch.rope_with_sin_cos_cache  mean time: 2.076254 ms
rope                           mean time: 0.408204 ms
rope_fp32x4                    mean time: 0.410063 ms
####################################################################################################
bs: 128, n: 1024, m: 256
torch                          mean time: 4.595430 ms
torch.rope_with_sin_cos_cache  mean time: 4.092623 ms
rope                           mean time: 0.821641 ms
rope_fp32x4                    mean time: 0.816536 ms
####################################################################################################
bs: 128, n: 2048, m: 128
torch                          mean time: 4.535651 ms
torch.rope_with_sin_cos_cache  mean time: 4.119844 ms
rope                           mean time: 0.816738 ms
rope_fp32x4                    mean time: 0.816076 ms
####################################################################################################
bs: 128, n: 2048, m: 256
torch                          mean time: 9.684505 ms
torch.rope_with_sin_cos_cache  mean time: 8.769187 ms
rope                           mean time: 1.669870 ms
rope_fp32x4                    mean time: 1.586758 ms
####################################################################################################
bs: 128, n: 4096, m: 128
torch                          mean time: 9.543169 ms
torch.rope_with_sin_cos_cache  mean time: 8.799918 ms
rope                           mean time: 1.648025 ms
rope_fp32x4                    mean time: 1.554470 ms
####################################################################################################
bs: 128, n: 4096, m: 256
torch                          mean time: 17.414704 ms
torch.rope_with_sin_cos_cache  mean time: 17.215360 ms
rope                           mean time: 3.218151 ms
rope_fp32x4                    mean time: 3.105632 ms
####################################################################################################
bs: 128, n: 8192, m: 128
torch                          mean time: 86.233111 ms
torch.rope_with_sin_cos_cache  mean time: 56.642383 ms
rope                           mean time: 3.187057 ms
rope_fp32x4                    mean time: 3.110710 ms
####################################################################################################
bs: 128, n: 8192, m: 256
torch                          mean time: 787.285911 ms
torch.rope_with_sin_cos_cache  mean time: 745.442381 ms
rope                           mean time: 123.269773 ms
rope_fp32x4                    mean time: 189.200739 ms
```
