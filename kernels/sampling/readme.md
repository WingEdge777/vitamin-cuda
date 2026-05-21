# LLM Generattion Sampling Kernels

- [ ] sampling_topk_topp_batched
- [ ] sampling_topk_topp_split_k

## Test

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

## Sample output

```yaml
####################################################################################################
bs: 1, vocab_size: 128000
torch                                    mean time: 0.293527 ms, 0.87 GB/s
flashinfer                               mean time: 0.114303 ms, speedup: 2.57, 2.24 GB/s
sampling_topk_topp_batched               mean time: 0.055709 ms, speedup: 5.27, 4.60 GB/s
####################################################################################################
bs: 1, vocab_size: 256000
torch                                    mean time: 0.397906 ms, 1.29 GB/s
flashinfer                               mean time: 0.159925 ms, speedup: 2.49, 3.20 GB/s
sampling_topk_topp_batched               mean time: 0.111866 ms, speedup: 3.56, 4.58 GB/s
####################################################################################################
bs: 1, vocab_size: 320000
torch                                    mean time: 0.530019 ms, 1.21 GB/s
flashinfer                               mean time: 0.144826 ms, speedup: 3.66, 4.42 GB/s
sampling_topk_topp_batched               mean time: 0.097341 ms, speedup: 5.45, 6.57 GB/s
####################################################################################################
bs: 8, vocab_size: 128000
torch                                    mean time: 0.746184 ms, 2.74 GB/s
flashinfer                               mean time: 0.107678 ms, speedup: 6.93, 19.02 GB/s
sampling_topk_topp_batched               mean time: 0.054862 ms, speedup: 13.60, 37.33 GB/s
####################################################################################################
bs: 8, vocab_size: 256000
torch                                    mean time: 1.266093 ms, 3.24 GB/s
flashinfer                               mean time: 0.186926 ms, speedup: 6.77, 21.91 GB/s
sampling_topk_topp_batched               mean time: 0.085241 ms, speedup: 14.85, 48.05 GB/s
####################################################################################################
bs: 8, vocab_size: 320000
torch                                    mean time: 1.588051 ms, 3.22 GB/s
flashinfer                               mean time: 0.232626 ms, speedup: 6.83, 22.01 GB/s
sampling_topk_topp_batched               mean time: 0.123562 ms, speedup: 12.85, 41.44 GB/s
####################################################################################################
bs: 16, vocab_size: 128000
torch                                    mean time: 1.094002 ms, 3.74 GB/s
flashinfer                               mean time: 0.187955 ms, speedup: 5.82, 21.79 GB/s
sampling_topk_topp_batched               mean time: 0.056973 ms, speedup: 19.20, 71.89 GB/s
####################################################################################################
bs: 16, vocab_size: 256000
torch                                    mean time: 2.276942 ms, 3.60 GB/s
flashinfer                               mean time: 0.375543 ms, speedup: 6.06, 21.81 GB/s
sampling_topk_topp_batched               mean time: 0.132943 ms, speedup: 17.13, 61.62 GB/s
####################################################################################################
bs: 16, vocab_size: 320000
torch                                    mean time: 2.960942 ms, 3.46 GB/s
flashinfer                               mean time: 0.388997 ms, speedup: 7.61, 26.32 GB/s
sampling_topk_topp_batched               mean time: 0.100191 ms, speedup: 29.55, 102.21 GB/s
####################################################################################################
bs: 32, vocab_size: 128000
torch                                    mean time: 2.173617 ms, 3.77 GB/s
flashinfer                               mean time: 0.289034 ms, speedup: 7.52, 28.34 GB/s
sampling_topk_topp_batched               mean time: 0.099781 ms, speedup: 21.78, 82.10 GB/s
####################################################################################################
bs: 32, vocab_size: 256000
torch                                    mean time: 4.557913 ms, 3.59 GB/s
flashinfer                               mean time: 0.638513 ms, speedup: 7.14, 25.66 GB/s
sampling_topk_topp_batched               mean time: 0.122755 ms, speedup: 37.13, 133.47 GB/s
####################################################################################################
bs: 32, vocab_size: 320000
torch                                    mean time: 5.527753 ms, 3.70 GB/s
flashinfer                               mean time: 0.835149 ms, speedup: 6.62, 24.52 GB/s
sampling_topk_topp_batched               mean time: 0.137315 ms, speedup: 40.26, 149.15 GB/s
```
