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
torch                          mean time: 0.338417 ms
torch.rope_with_sin_cos_cache  mean time: 0.062025 ms
rope                           mean time: 0.013051 ms
####################################################################################################
n: 512, m: 256
torch                          mean time: 0.395294 ms
torch.rope_with_sin_cos_cache  mean time: 0.060028 ms
rope                           mean time: 0.010533 ms
####################################################################################################
n: 1024, m: 128
torch                          mean time: 0.314019 ms
torch.rope_with_sin_cos_cache  mean time: 0.075795 ms
rope                           mean time: 0.014411 ms
####################################################################################################
n: 1024, m: 256
torch                          mean time: 0.309554 ms
torch.rope_with_sin_cos_cache  mean time: 0.067735 ms
rope                           mean time: 0.012363 ms
####################################################################################################
n: 2048, m: 128
torch                          mean time: 0.311622 ms
torch.rope_with_sin_cos_cache  mean time: 0.072086 ms
rope                           mean time: 0.020575 ms
####################################################################################################
n: 2048, m: 256
torch                          mean time: 0.396738 ms
torch.rope_with_sin_cos_cache  mean time: 0.068628 ms
rope                           mean time: 0.011369 ms
####################################################################################################
n: 4096, m: 128
torch                          mean time: 0.306962 ms
torch.rope_with_sin_cos_cache  mean time: 0.058664 ms
rope                           mean time: 0.021087 ms
####################################################################################################
n: 4096, m: 256
torch                          mean time: 0.442126 ms
torch.rope_with_sin_cos_cache  mean time: 0.086117 ms
rope                           mean time: 0.015870 ms
####################################################################################################
n: 8192, m: 128
torch                          mean time: 0.336158 ms
torch.rope_with_sin_cos_cache  mean time: 0.111435 ms
rope                           mean time: 0.022904 ms
####################################################################################################
n: 8192, m: 256
torch                          mean time: 0.560861 ms
torch.rope_with_sin_cos_cache  mean time: 0.284379 ms
rope                           mean time: 0.025427 ms
```
