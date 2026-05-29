# LLM Generattion Sampling Kernels

Mainly use `array insertion sort && merge`

- [x] sampling_topk_topp_batched
- [x] sampling_topk_topp_split_k

## Test

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

## Sample output

### benchmark with cold L2 cache

```yaml
GPU: NVIDIA GeForce RTX 5060 Laptop GPU  fi=0.6.9
timer = CUDA events, no graph, no CUPTI; only cold_l2_cache toggled

config        kernel            warmL2(cold=F)  coldL2(cold=T)  cold/warm    cold speedup
####################################################################################################
bs1/128k      flashinfer                0.0857          0.0877       1.02x
bs1/128k      custom_batched            0.0580          0.0911       1.57x       0.96x
bs1/128k      custom_splitk             0.0357          0.0656       1.84x       1.34x
####################################################################################################
bs1/256k      flashinfer                0.1351          0.1381       1.02x
bs1/256k      custom_batched            0.0876          0.1348       1.54x       1.02x
bs1/256k      custom_splitk             0.0368          0.0671       1.82x       2.06x
####################################################################################################
bs1/320k      flashinfer                0.1538          0.1575       1.02x
bs1/320k      custom_batched            0.1018          0.1577       1.55x       1.00x
bs1/320k      custom_splitk             0.0504          0.0726       1.44x       2.17x
####################################################################################################
bs4/128k      flashinfer                0.0916          0.0952       1.04x
bs4/128k      custom_batched            0.0568          0.0843       1.48x       1.13x
bs4/128k      custom_splitk             0.0482          0.0671       1.39x       1.42x
####################################################################################################
bs4/256k      flashinfer                0.1530          0.1601       1.05x
bs4/256k      custom_batched            0.0854          0.1282       1.50x       1.25x
bs4/256k      custom_splitk             0.0546          0.0737       1.35x       2.17x
####################################################################################################
bs4/320k      flashinfer                0.1971          0.2079       1.06x
bs4/320k      custom_batched            0.0996          0.1472       1.48x       1.41x
bs4/320k      custom_splitk             0.0549          0.0776       1.41x       2.68x
####################################################################################################
bs8/128k      flashinfer                0.1147          0.1216       1.06x
bs8/128k      custom_batched            0.0568          0.0861       1.52x       1.41x
bs8/128k      custom_splitk             0.0696          0.0909       1.31x       1.34x
####################################################################################################
bs8/256k      flashinfer                0.1839          0.1986       1.08x
bs8/256k      custom_batched            0.0883          0.1318       1.49x       1.51x
bs8/256k      custom_splitk             0.0793          0.1026       1.29x       1.94x
####################################################################################################
bs8/320k      flashinfer                0.2421          0.2646       1.09x
bs8/320k      custom_batched            0.1016          0.1513       1.49x       1.75x
bs8/320k      custom_splitk             0.0867          0.1109       1.28x       2.39x
####################################################################################################
bs16/128k     flashinfer                0.1324          0.1451       1.10x
bs16/128k     custom_batched            0.0572          0.0869       1.52x       1.67x
####################################################################################################
bs16/256k     flashinfer                0.2537          0.2585       1.02x
bs16/256k     custom_batched            0.0885          0.1488       1.68x       1.74x
####################################################################################################
bs16/320k     flashinfer                0.3529          0.3641       1.03x
bs16/320k     custom_batched            0.1018          0.1682       1.65x       2.16x
####################################################################################################
bs32/128k     flashinfer                0.2495          0.2597       1.04x
bs32/128k     custom_batched            0.0818          0.0958       1.17x       2.71x
####################################################################################################
bs32/256k     flashinfer                0.6082          0.6365       1.05x
bs32/256k     custom_batched            0.1337          0.1497       1.12x       4.25x
####################################################################################################
bs32/320k     flashinfer                0.7970          0.8069       1.01x
bs32/320k     custom_batched            0.1460          0.1734       1.19x       4.65x
```

### random spike data

```yaml
####################################################################################################
bs: 1, vocab_size: 128000
torch                                    mean time: 0.537336 ms, 0.48 GB/s
flashinfer                               mean time: 0.081936 ms, speedup: 6.56, 3.12 GB/s
sampling_topk_topp_batched               mean time: 0.055265 ms, speedup: 9.72, 4.63 GB/s
sampling_topk_topp_split_k               mean time: 0.038957 ms, speedup: 13.79, 6.57 GB/s
####################################################################################################
bs: 1, vocab_size: 256000
torch                                    mean time: 0.519055 ms, 0.99 GB/s
flashinfer                               mean time: 0.140417 ms, speedup: 3.70, 3.65 GB/s
sampling_topk_topp_batched               mean time: 0.084192 ms, speedup: 6.17, 6.08 GB/s
sampling_topk_topp_split_k               mean time: 0.036327 ms, speedup: 14.29, 14.09 GB/s
####################################################################################################
bs: 1, vocab_size: 320000
torch                                    mean time: 0.540429 ms, 1.18 GB/s
flashinfer                               mean time: 0.161556 ms, speedup: 3.35, 3.96 GB/s
sampling_topk_topp_batched               mean time: 0.095184 ms, speedup: 5.68, 6.72 GB/s
sampling_topk_topp_split_k               mean time: 0.037684 ms, speedup: 14.34, 16.98 GB/s
####################################################################################################
bs: 4, vocab_size: 128000
torch                                    mean time: 0.616612 ms, 1.66 GB/s
flashinfer                               mean time: 0.086498 ms, speedup: 7.13, 11.84 GB/s
sampling_topk_topp_batched               mean time: 0.055023 ms, speedup: 11.21, 18.61 GB/s
sampling_topk_topp_split_k               mean time: 0.048628 ms, speedup: 12.68, 21.06 GB/s
####################################################################################################
bs: 4, vocab_size: 256000
torch                                    mean time: 0.931264 ms, 2.20 GB/s
flashinfer                               mean time: 0.154697 ms, speedup: 6.02, 13.24 GB/s
sampling_topk_topp_batched               mean time: 0.084510 ms, speedup: 11.02, 24.23 GB/s
sampling_topk_topp_split_k               mean time: 0.051476 ms, speedup: 18.09, 39.79 GB/s
####################################################################################################
bs: 4, vocab_size: 320000
torch                                    mean time: 1.104376 ms, 2.32 GB/s
flashinfer                               mean time: 0.231562 ms, speedup: 4.77, 11.06 GB/s
sampling_topk_topp_batched               mean time: 0.097490 ms, speedup: 11.33, 26.26 GB/s
sampling_topk_topp_split_k               mean time: 0.050732 ms, speedup: 21.77, 50.46 GB/s
####################################################################################################
bs: 8, vocab_size: 128000
torch                                    mean time: 0.797353 ms, 2.57 GB/s
flashinfer                               mean time: 0.105724 ms, speedup: 7.54, 19.37 GB/s
sampling_topk_topp_batched               mean time: 0.083064 ms, speedup: 9.60, 24.66 GB/s
sampling_topk_topp_split_k               mean time: 0.065168 ms, speedup: 12.24, 31.43 GB/s
####################################################################################################
bs: 8, vocab_size: 256000
torch                                    mean time: 1.296351 ms, 3.16 GB/s
flashinfer                               mean time: 0.182281 ms, speedup: 7.11, 22.47 GB/s
sampling_topk_topp_batched               mean time: 0.084636 ms, speedup: 15.32, 48.40 GB/s
sampling_topk_topp_split_k               mean time: 0.075195 ms, speedup: 17.24, 54.47 GB/s
####################################################################################################
bs: 8, vocab_size: 320000
torch                                    mean time: 1.579942 ms, 3.24 GB/s
flashinfer                               mean time: 0.267332 ms, speedup: 5.91, 19.15 GB/s
sampling_topk_topp_batched               mean time: 0.098516 ms, speedup: 16.04, 51.97 GB/s
sampling_topk_topp_split_k               mean time: 0.078897 ms, speedup: 20.03, 64.89 GB/s
####################################################################################################
bs: 16, vocab_size: 128000
torch                                    mean time: 1.102630 ms, 3.71 GB/s
flashinfer                               mean time: 0.135733 ms, speedup: 8.12, 30.18 GB/s
sampling_topk_topp_batched               mean time: 0.057381 ms, speedup: 19.22, 71.38 GB/s
####################################################################################################
bs: 16, vocab_size: 256000
torch                                    mean time: 2.317547 ms, 3.53 GB/s
flashinfer                               mean time: 0.307686 ms, speedup: 7.53, 26.62 GB/s
sampling_topk_topp_batched               mean time: 0.102009 ms, speedup: 22.72, 80.31 GB/s
####################################################################################################
bs: 16, vocab_size: 320000
torch                                    mean time: 3.069562 ms, 3.34 GB/s
flashinfer                               mean time: 0.364070 ms, speedup: 8.43, 28.13 GB/s
sampling_topk_topp_batched               mean time: 0.126470 ms, speedup: 24.27, 80.97 GB/s
####################################################################################################
bs: 32, vocab_size: 128000
torch                                    mean time: 2.197257 ms, 3.73 GB/s
flashinfer                               mean time: 0.266425 ms, speedup: 8.25, 30.75 GB/s
sampling_topk_topp_batched               mean time: 0.076456 ms, speedup: 28.74, 107.15 GB/s
####################################################################################################
bs: 32, vocab_size: 256000
torch                                    mean time: 4.632301 ms, 3.54 GB/s
flashinfer                               mean time: 0.598686 ms, speedup: 7.74, 27.37 GB/s
sampling_topk_topp_batched               mean time: 0.127663 ms, speedup: 36.29, 128.34 GB/s
####################################################################################################
bs: 32, vocab_size: 320000
torch                                    mean time: 5.643508 ms, 3.63 GB/s
flashinfer                               mean time: 0.774221 ms, speedup: 7.29, 26.45 GB/s
sampling_topk_topp_batched               mean time: 0.137926 ms, speedup: 40.92, 148.49 GB/s
```

### real qwen3-8b logits data

```yaml
####################################################################################################
bs: 1, vocab_size: 151936
torch                                    mean time: 0.333807 ms, 0.91 GB/s
flashinfer                               mean time: 0.141696 ms, speedup: 2.36, 2.14 GB/s
sampling_topk_topp_batched               mean time: 0.075829 ms, speedup: 4.40, 4.01 GB/s
sampling_topk_topp_split_k               mean time: 0.031688 ms, speedup: 10.53, 9.59 GB/s
####################################################################################################
bs: 1, vocab_size: 151936
torch                                    mean time: 0.385421 ms, 0.79 GB/s
flashinfer                               mean time: 0.081062 ms, speedup: 4.75, 3.75 GB/s
sampling_topk_topp_batched               mean time: 0.077032 ms, speedup: 5.00, 3.94 GB/s
sampling_topk_topp_split_k               mean time: 0.031270 ms, speedup: 12.33, 9.72 GB/s
####################################################################################################
bs: 1, vocab_size: 151936
torch                                    mean time: 0.332377 ms, 0.91 GB/s
flashinfer                               mean time: 0.081582 ms, speedup: 4.07, 3.72 GB/s
sampling_topk_topp_batched               mean time: 0.055480 ms, speedup: 5.99, 5.48 GB/s
sampling_topk_topp_split_k               mean time: 0.032620 ms, speedup: 10.19, 9.32 GB/s
####################################################################################################
bs: 1, vocab_size: 151936
torch                                    mean time: 0.385464 ms, 0.79 GB/s
flashinfer                               mean time: 0.106221 ms, speedup: 3.63, 2.86 GB/s
sampling_topk_topp_batched               mean time: 0.048918 ms, speedup: 7.88, 6.21 GB/s
sampling_topk_topp_split_k               mean time: 0.033564 ms, speedup: 11.48, 9.05 GB/s
####################################################################################################
bs: 1, vocab_size: 151936
torch                                    mean time: 0.368339 ms, 0.82 GB/s
flashinfer                               mean time: 0.102984 ms, speedup: 3.58, 2.95 GB/s
sampling_topk_topp_batched               mean time: 0.049881 ms, speedup: 7.38, 6.09 GB/s
sampling_topk_topp_split_k               mean time: 0.031515 ms, speedup: 11.69, 9.64 GB/s
####################################################################################################
bs: 4, vocab_size: 151936
torch                                    mean time: 0.706000 ms, 1.72 GB/s
flashinfer                               mean time: 0.089564 ms, speedup: 7.88, 13.57 GB/s
sampling_topk_topp_batched               mean time: 0.052516 ms, speedup: 13.44, 23.15 GB/s
sampling_topk_topp_split_k               mean time: 0.054109 ms, speedup: 13.05, 22.46 GB/s
####################################################################################################
bs: 4, vocab_size: 151936
torch                                    mean time: 0.703697 ms, 1.73 GB/s
flashinfer                               mean time: 0.132022 ms, speedup: 5.33, 9.21 GB/s
sampling_topk_topp_batched               mean time: 0.051942 ms, speedup: 13.55, 23.40 GB/s
sampling_topk_topp_split_k               mean time: 0.044773 ms, speedup: 15.72, 27.15 GB/s
####################################################################################################
bs: 4, vocab_size: 151936
torch                                    mean time: 0.740746 ms, 1.64 GB/s
flashinfer                               mean time: 0.091355 ms, speedup: 8.11, 13.31 GB/s
sampling_topk_topp_batched               mean time: 0.054907 ms, speedup: 13.49, 22.14 GB/s
sampling_topk_topp_split_k               mean time: 0.045861 ms, speedup: 16.15, 26.50 GB/s
####################################################################################################
bs: 4, vocab_size: 151936
torch                                    mean time: 0.719965 ms, 1.69 GB/s
flashinfer                               mean time: 0.086940 ms, speedup: 8.28, 13.98 GB/s
sampling_topk_topp_batched               mean time: 0.049303 ms, speedup: 14.60, 24.65 GB/s
sampling_topk_topp_split_k               mean time: 0.045837 ms, speedup: 15.71, 26.52 GB/s
####################################################################################################
bs: 4, vocab_size: 151936
torch                                    mean time: 0.731304 ms, 1.66 GB/s
flashinfer                               mean time: 0.087465 ms, speedup: 8.36, 13.90 GB/s
sampling_topk_topp_batched               mean time: 0.050458 ms, speedup: 14.49, 24.09 GB/s
sampling_topk_topp_split_k               mean time: 0.044984 ms, speedup: 16.26, 27.02 GB/s
####################################################################################################
bs: 16, vocab_size: 151936
torch                                    mean time: 1.341189 ms, 3.63 GB/s
flashinfer                               mean time: 0.145224 ms, speedup: 9.24, 33.48 GB/s
sampling_topk_topp_batched               mean time: 0.058288 ms, speedup: 23.01, 83.41 GB/s
####################################################################################################
bs: 16, vocab_size: 151936
torch                                    mean time: 1.320675 ms, 3.68 GB/s
flashinfer                               mean time: 0.145425 ms, speedup: 9.08, 33.43 GB/s
sampling_topk_topp_batched               mean time: 0.059137 ms, speedup: 22.33, 82.22 GB/s
####################################################################################################
bs: 16, vocab_size: 151936
torch                                    mean time: 1.351302 ms, 3.60 GB/s
flashinfer                               mean time: 0.163167 ms, speedup: 8.28, 29.80 GB/s
sampling_topk_topp_batched               mean time: 0.062388 ms, speedup: 21.66, 77.93 GB/s
####################################################################################################
bs: 16, vocab_size: 151936
torch                                    mean time: 1.396582 ms, 3.48 GB/s
flashinfer                               mean time: 0.177260 ms, speedup: 7.88, 27.43 GB/s
sampling_topk_topp_batched               mean time: 0.057654 ms, speedup: 24.22, 84.33 GB/s
####################################################################################################
bs: 16, vocab_size: 151936
torch                                    mean time: 1.429869 ms, 3.40 GB/s
flashinfer                               mean time: 0.172361 ms, speedup: 8.30, 28.21 GB/s
sampling_topk_topp_batched               mean time: 0.062052 ms, speedup: 23.04, 78.35 GB/s
####################################################################################################
bs: 32, vocab_size: 151936
torch                                    mean time: 2.702709 ms, 3.60 GB/s
flashinfer                               mean time: 0.285007 ms, speedup: 9.48, 34.12 GB/s
sampling_topk_topp_batched               mean time: 0.093435 ms, speedup: 28.93, 104.07 GB/s
####################################################################################################
bs: 32, vocab_size: 151936
torch                                    mean time: 2.574061 ms, 3.78 GB/s
flashinfer                               mean time: 0.301973 ms, speedup: 8.52, 32.20 GB/s
sampling_topk_topp_batched               mean time: 0.072180 ms, speedup: 35.66, 134.72 GB/s
####################################################################################################
bs: 32, vocab_size: 151936
torch                                    mean time: 2.724618 ms, 3.57 GB/s
flashinfer                               mean time: 0.337126 ms, speedup: 8.08, 28.84 GB/s
sampling_topk_topp_batched               mean time: 0.089834 ms, speedup: 30.33, 108.24 GB/s
####################################################################################################
bs: 32, vocab_size: 151936
torch                                    mean time: 2.636955 ms, 3.69 GB/s
flashinfer                               mean time: 0.300759 ms, speedup: 8.77, 32.33 GB/s
sampling_topk_topp_batched               mean time: 0.065455 ms, speedup: 40.29, 148.56 GB/s
####################################################################################################
bs: 32, vocab_size: 151936
torch                                    mean time: 2.872458 ms, 3.39 GB/s
flashinfer                               mean time: 0.356981 ms, speedup: 8.05, 27.24 GB/s
sampling_topk_topp_batched               mean time: 0.070633 ms, speedup: 40.67, 137.67 GB/s
```