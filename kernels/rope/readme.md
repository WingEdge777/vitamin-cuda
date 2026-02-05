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
n: 512, m: 128
torch                          mean time: 0.344705 ms
torch.rope_with_sin_cos_cache  mean time: 0.074115 ms
rope                           mean time: 0.015619 ms
rope_fp32x4                    mean time: 0.008188 ms
####################################################################################################
n: 512, m: 256
torch                          mean time: 0.383128 ms
torch.rope_with_sin_cos_cache  mean time: 0.055858 ms
rope                           mean time: 0.011565 ms
rope_fp32x4                    mean time: 0.014199 ms
####################################################################################################
n: 1024, m: 128
torch                          mean time: 0.261452 ms
torch.rope_with_sin_cos_cache  mean time: 0.053360 ms
rope                           mean time: 0.014579 ms
rope_fp32x4                    mean time: 0.008430 ms
####################################################################################################
n: 1024, m: 256
torch                          mean time: 0.248948 ms
torch.rope_with_sin_cos_cache  mean time: 0.048255 ms
rope                           mean time: 0.009767 ms
rope_fp32x4                    mean time: 0.010090 ms
####################################################################################################
n: 2048, m: 128
torch                          mean time: 0.285137 ms
torch.rope_with_sin_cos_cache  mean time: 0.047323 ms
rope                           mean time: 0.009408 ms
rope_fp32x4                    mean time: 0.014790 ms
####################################################################################################
n: 2048, m: 256
torch                          mean time: 0.292945 ms
torch.rope_with_sin_cos_cache  mean time: 0.053592 ms
rope                           mean time: 0.010897 ms
rope_fp32x4                    mean time: 0.011582 ms
####################################################################################################
n: 4096, m: 128
torch                          mean time: 0.286234 ms
torch.rope_with_sin_cos_cache  mean time: 0.062171 ms
rope                           mean time: 0.012274 ms
rope_fp32x4                    mean time: 0.015684 ms
####################################################################################################
n: 4096, m: 256
torch                          mean time: 0.283420 ms
torch.rope_with_sin_cos_cache  mean time: 0.080113 ms
rope                           mean time: 0.015868 ms
rope_fp32x4                    mean time: 0.017967 ms
####################################################################################################
n: 8192, m: 128
torch                          mean time: 0.455298 ms
torch.rope_with_sin_cos_cache  mean time: 0.082223 ms
rope                           mean time: 0.020271 ms
rope_fp32x4                    mean time: 0.011539 ms
####################################################################################################
n: 8192, m: 256
torch                          mean time: 0.565554 ms
torch.rope_with_sin_cos_cache  mean time: 0.275599 ms
rope                           mean time: 0.025792 ms
rope_fp32x4                    mean time: 0.014390 ms
```
