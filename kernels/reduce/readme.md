# reduce_sum

## 说明

reduce_sum kernel

- [x] reduce_sum fp32/fp16 版
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
torch                          mean time: 0.066859 ms
reduce_sum                     mean time: 0.036893 ms
torch                          mean time: 0.018864 ms
reduce_sum_half                mean time: 0.038802 ms
####################################################################################################
n: 1024, m: 2048
torch                          mean time: 0.058253 ms
reduce_sum                     mean time: 0.035388 ms
torch                          mean time: 0.023238 ms
reduce_sum_half                mean time: 0.040391 ms
####################################################################################################
n: 1024, m: 4096
torch                          mean time: 0.027540 ms
reduce_sum                     mean time: 0.038788 ms
torch                          mean time: 0.021334 ms
reduce_sum_half                mean time: 0.041792 ms
####################################################################################################
n: 2048, m: 1024
torch                          mean time: 0.030639 ms
reduce_sum                     mean time: 0.040931 ms
torch                          mean time: 0.033329 ms
reduce_sum_half                mean time: 0.052609 ms
####################################################################################################
n: 2048, m: 2048
torch                          mean time: 0.026194 ms
reduce_sum                     mean time: 0.038825 ms
torch                          mean time: 0.026860 ms
reduce_sum_half                mean time: 0.049642 ms
####################################################################################################
n: 2048, m: 4096
torch                          mean time: 0.029246 ms
reduce_sum                     mean time: 0.071709 ms
torch                          mean time: 0.033398 ms
reduce_sum_half                mean time: 0.077679 ms
####################################################################################################
n: 4096, m: 1024
torch                          mean time: 0.043786 ms
reduce_sum                     mean time: 0.033638 ms
torch                          mean time: 0.023280 ms
reduce_sum_half                mean time: 0.042062 ms
####################################################################################################
n: 4096, m: 2048
torch                          mean time: 0.037445 ms
reduce_sum                     mean time: 0.062737 ms
torch                          mean time: 0.040068 ms
reduce_sum_half                mean time: 0.069430 ms
####################################################################################################
n: 4096, m: 4096
torch                          mean time: 0.624860 ms
reduce_sum                     mean time: 0.825760 ms
torch                          mean time: 0.110296 ms
reduce_sum_half                mean time: 0.187578 ms
```
