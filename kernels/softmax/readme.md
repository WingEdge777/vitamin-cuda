# softmax

## 说明

softmax kernel

- [x] safe online softmax fp32/fp16 版
- [x] safe online softmax fp32x4 版 (fp32向量化)
- [x] safe online softmax fp16x8 版 (fp16向量化, packed r/w)
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
