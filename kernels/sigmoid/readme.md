# Sigmoid

## 说明

sigmoid kernel

- [x] sigmoid fp32/fp16 版
- [x] sigmoid_fp16x2(fp16向量化)
- [x] sigmoid_fp16x8(fp16向量化)
- [x] sigmoid_fp16x8(fp16向量化, packed r/w)
- [x] pytorch op bindings && diff check

## 测试

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### 输出

```bash
####################################################################################################
n: 1024, m: 1024
torch                          mean time: 0.029449 ms
sigmoid                        mean time: 0.028635 ms
sigmoid_fp32x4                 mean time: 0.010126 ms
torch                          mean time: 0.016302 ms
sigmoid_half                   mean time: 0.015223 ms
sigmoid_fp16x2                 mean time: 0.015981 ms
sigmoid_fp16x8                 mean time: 0.012098 ms
sigmoid_fp16x8_packed          mean time: 0.015574 ms
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.018414 ms
sigmoid                        mean time: 0.029108 ms
sigmoid_fp32x4                 mean time: 0.013150 ms
torch                          mean time: 0.013879 ms
sigmoid_half                   mean time: 0.031012 ms
sigmoid_fp16x2                 mean time: 0.017127 ms
sigmoid_fp16x8                 mean time: 0.028135 ms
sigmoid_fp16x8_packed          mean time: 0.018357 ms
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.038361 ms
sigmoid                        mean time: 0.058420 ms
sigmoid_fp32x4                 mean time: 0.038400 ms
torch                          mean time: 0.026921 ms
sigmoid_half                   mean time: 0.061567 ms
sigmoid_fp16x2                 mean time: 0.034572 ms
sigmoid_fp16x8                 mean time: 0.044389 ms
sigmoid_fp16x8_packed          mean time: 0.026405 ms
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.014946 ms
sigmoid                        mean time: 0.028870 ms
sigmoid_fp32x4                 mean time: 0.013629 ms
torch                          mean time: 0.017595 ms
sigmoid_half                   mean time: 0.029063 ms
sigmoid_fp16x2                 mean time: 0.015894 ms
sigmoid_fp16x8                 mean time: 0.026175 ms
sigmoid_fp16x8_packed          mean time: 0.018562 ms
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.025935 ms
sigmoid                        mean time: 0.065047 ms
sigmoid_fp32x4                 mean time: 0.034529 ms
torch                          mean time: 0.021679 ms
sigmoid_half                   mean time: 0.059793 ms
sigmoid_fp16x2                 mean time: 0.029605 ms
sigmoid_fp16x8                 mean time: 0.040708 ms
sigmoid_fp16x8_packed          mean time: 0.027757 ms
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.218718 ms
sigmoid                        mean time: 0.218397 ms
sigmoid_fp32x4                 mean time: 0.212876 ms
torch                          mean time: 0.053832 ms
sigmoid_half                   mean time: 0.164681 ms
sigmoid_fp16x2                 mean time: 0.069732 ms
sigmoid_fp16x8                 mean time: 0.106206 ms
sigmoid_fp16x8_packed          mean time: 0.046554 ms
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.037825 ms
sigmoid                        mean time: 0.060588 ms
sigmoid_fp32x4                 mean time: 0.040090 ms
torch                          mean time: 0.026480 ms
sigmoid_half                   mean time: 0.058943 ms
sigmoid_fp16x2                 mean time: 0.031711 ms
sigmoid_fp16x8                 mean time: 0.042393 ms
sigmoid_fp16x8_packed          mean time: 0.020381 ms
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.212433 ms
sigmoid                        mean time: 0.215209 ms
sigmoid_fp32x4                 mean time: 0.210652 ms
torch                          mean time: 0.041770 ms
sigmoid_half                   mean time: 0.135674 ms
sigmoid_fp16x2                 mean time: 0.083758 ms
sigmoid_fp16x8                 mean time: 0.105754 ms
sigmoid_fp16x8_packed          mean time: 0.049772 ms
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.429733 ms
sigmoid                        mean time: 0.434803 ms
sigmoid_fp32x4                 mean time: 0.414464 ms
torch                          mean time: 0.211792 ms
sigmoid_half                   mean time: 0.426866 ms
sigmoid_fp16x2                 mean time: 0.241196 ms
sigmoid_fp16x8                 mean time: 0.234643 ms
sigmoid_fp16x8_packed          mean time: 0.211962 ms
```
