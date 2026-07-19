# cumsum

## Overview

cumsum kernels.

- [x] naive Torch cumsum
- [x] cumsum — FP32
- [x] cumsum — FP32x4
- [x] cumsum — FP32x4 multi-CTA scan (tid0 decoupled look-back)
- [x] pytorch op bindings && diff check

## Run tests

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

### Sample output

```bash
####################################################################################################
n: 1, m: 2048
torch                          mean time: 0.011056 ms
cumsum_fp32                    mean time: 0.011184 ms, speedup: 0.99
cumsum_fp32x4                  mean time: 0.007696 ms, speedup: 1.44
cumsum_fp32x4_multi_cta_scan   mean time: 0.009392 ms, speedup: 1.18
####################################################################################################
n: 1, m: 4096
torch                          mean time: 0.010848 ms
cumsum_fp32                    mean time: 0.016336 ms, speedup: 0.66
cumsum_fp32x4                  mean time: 0.009184 ms, speedup: 1.18
cumsum_fp32x4_multi_cta_scan   mean time: 0.009152 ms, speedup: 1.19
####################################################################################################
n: 1, m: 8192
torch                          mean time: 0.010016 ms
cumsum_fp32                    mean time: 0.025440 ms, speedup: 0.39
cumsum_fp32x4                  mean time: 0.011472 ms, speedup: 0.87
cumsum_fp32x4_multi_cta_scan   mean time: 0.009248 ms, speedup: 1.08
####################################################################################################
n: 1, m: 12800
torch                          mean time: 0.010880 ms
cumsum_fp32                    mean time: 0.036096 ms, speedup: 0.30
cumsum_fp32x4                  mean time: 0.015760 ms, speedup: 0.69
cumsum_fp32x4_multi_cta_scan   mean time: 0.009408 ms, speedup: 1.16
####################################################################################################
n: 1, m: 32768
torch                          mean time: 0.011840 ms
cumsum_fp32                    mean time: 0.084192 ms, speedup: 0.14
cumsum_fp32x4                  mean time: 0.027520 ms, speedup: 0.43
cumsum_fp32x4_multi_cta_scan   mean time: 0.009776 ms, speedup: 1.21
####################################################################################################
n: 1, m: 65536
torch                          mean time: 0.012112 ms
cumsum_fp32                    mean time: 0.165056 ms, speedup: 0.07
cumsum_fp32x4                  mean time: 0.047632 ms, speedup: 0.25
cumsum_fp32x4_multi_cta_scan   mean time: 0.009936 ms, speedup: 1.22
####################################################################################################
n: 32, m: 2048
torch                          mean time: 0.016432 ms
cumsum_fp32                    mean time: 0.012256 ms, speedup: 1.34
cumsum_fp32x4                  mean time: 0.009152 ms, speedup: 1.80
cumsum_fp32x4_multi_cta_scan   mean time: 0.010016 ms, speedup: 1.64
####################################################################################################
n: 32, m: 4096
torch                          mean time: 0.017520 ms
cumsum_fp32                    mean time: 0.017296 ms, speedup: 1.01
cumsum_fp32x4                  mean time: 0.011328 ms, speedup: 1.55
cumsum_fp32x4_multi_cta_scan   mean time: 0.011360 ms, speedup: 1.54
####################################################################################################
n: 32, m: 8192
torch                          mean time: 0.027840 ms
cumsum_fp32                    mean time: 0.028400 ms, speedup: 0.98
cumsum_fp32x4                  mean time: 0.016848 ms, speedup: 1.65
cumsum_fp32x4_multi_cta_scan   mean time: 0.014288 ms, speedup: 1.95
####################################################################################################
n: 32, m: 12800
torch                          mean time: 0.029568 ms
cumsum_fp32                    mean time: 0.043376 ms, speedup: 0.68
cumsum_fp32x4                  mean time: 0.021696 ms, speedup: 1.36
cumsum_fp32x4_multi_cta_scan   mean time: 0.016032 ms, speedup: 1.84
####################################################################################################
n: 32, m: 32768
torch                          mean time: 0.061888 ms
cumsum_fp32                    mean time: 0.091152 ms, speedup: 0.68
cumsum_fp32x4                  mean time: 0.045968 ms, speedup: 1.35
cumsum_fp32x4_multi_cta_scan   mean time: 0.029936 ms, speedup: 2.07
####################################################################################################
n: 32, m: 65536
torch                          mean time: 0.116384 ms
cumsum_fp32                    mean time: 0.184848 ms, speedup: 0.63
cumsum_fp32x4                  mean time: 0.082784 ms, speedup: 1.41
cumsum_fp32x4_multi_cta_scan   mean time: 0.053600 ms, speedup: 2.17
####################################################################################################
n: 64, m: 2048
torch                          mean time: 0.017328 ms
cumsum_fp32                    mean time: 0.013232 ms, speedup: 1.31
cumsum_fp32x4                  mean time: 0.010032 ms, speedup: 1.73
cumsum_fp32x4_multi_cta_scan   mean time: 0.011584 ms, speedup: 1.50
####################################################################################################
n: 64, m: 4096
torch                          mean time: 0.027680 ms
cumsum_fp32                    mean time: 0.019808 ms, speedup: 1.40
cumsum_fp32x4                  mean time: 0.013072 ms, speedup: 2.12
cumsum_fp32x4_multi_cta_scan   mean time: 0.013472 ms, speedup: 2.05
####################################################################################################
n: 64, m: 8192
torch                          mean time: 0.033936 ms
cumsum_fp32                    mean time: 0.033168 ms, speedup: 1.02
cumsum_fp32x4                  mean time: 0.018720 ms, speedup: 1.81
cumsum_fp32x4_multi_cta_scan   mean time: 0.016592 ms, speedup: 2.05
####################################################################################################
n: 64, m: 12800
torch                          mean time: 0.048176 ms
cumsum_fp32                    mean time: 0.046992 ms, speedup: 1.03
cumsum_fp32x4                  mean time: 0.025312 ms, speedup: 1.90
cumsum_fp32x4_multi_cta_scan   mean time: 0.023712 ms, speedup: 2.03
####################################################################################################
n: 64, m: 32768
torch                          mean time: 0.081408 ms
cumsum_fp32                    mean time: 0.107440 ms, speedup: 0.76
cumsum_fp32x4                  mean time: 0.062144 ms, speedup: 1.31
cumsum_fp32x4_multi_cta_scan   mean time: 0.054560 ms, speedup: 1.49
####################################################################################################
n: 64, m: 65536
torch                          mean time: 0.152480 ms
cumsum_fp32                    mean time: 0.203120 ms, speedup: 0.75
cumsum_fp32x4                  mean time: 0.126528 ms, speedup: 1.21
cumsum_fp32x4_multi_cta_scan   mean time: 0.108672 ms, speedup: 1.40
####################################################################################################
n: 128, m: 2048
torch                          mean time: 0.026128 ms
cumsum_fp32                    mean time: 0.015552 ms, speedup: 1.68
cumsum_fp32x4                  mean time: 0.010768 ms, speedup: 2.43
cumsum_fp32x4_multi_cta_scan   mean time: 0.013472 ms, speedup: 1.94
####################################################################################################
n: 128, m: 4096
torch                          mean time: 0.032368 ms
cumsum_fp32                    mean time: 0.025184 ms, speedup: 1.29
cumsum_fp32x4                  mean time: 0.013392 ms, speedup: 2.42
cumsum_fp32x4_multi_cta_scan   mean time: 0.016608 ms, speedup: 1.95
####################################################################################################
n: 128, m: 8192
torch                          mean time: 0.057344 ms
cumsum_fp32                    mean time: 0.043472 ms, speedup: 1.32
cumsum_fp32x4                  mean time: 0.026768 ms, speedup: 2.14
cumsum_fp32x4_multi_cta_scan   mean time: 0.029232 ms, speedup: 1.96
####################################################################################################
n: 128, m: 12800
torch                          mean time: 0.062736 ms
cumsum_fp32                    mean time: 0.061248 ms, speedup: 1.02
cumsum_fp32x4                  mean time: 0.050608 ms, speedup: 1.24
cumsum_fp32x4_multi_cta_scan   mean time: 0.043776 ms, speedup: 1.43
####################################################################################################
n: 128, m: 32768
torch                          mean time: 0.146528 ms
cumsum_fp32                    mean time: 0.140800 ms, speedup: 1.04
cumsum_fp32x4                  mean time: 0.113904 ms, speedup: 1.29
cumsum_fp32x4_multi_cta_scan   mean time: 0.108976 ms, speedup: 1.34
####################################################################################################
n: 128, m: 65536
torch                          mean time: 0.266880 ms
cumsum_fp32                    mean time: 0.254416 ms, speedup: 1.05
cumsum_fp32x4                  mean time: 0.214752 ms, speedup: 1.24
cumsum_fp32x4_multi_cta_scan   mean time: 0.215824 ms, speedup: 1.24
```
