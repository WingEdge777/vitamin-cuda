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
