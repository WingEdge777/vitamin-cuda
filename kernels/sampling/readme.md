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
torch                                    mean time: 0.297040 ms, 0.86 GB/s
flashinfer                               mean time: 0.080430 ms, speedup: 3.69, 3.18 GB/s
sampling_topk_topp_batched               mean time: 0.077968 ms, speedup: 3.81, 3.28 GB/s
####################################################################################################
bs: 1, vocab_size: 256000
torch                                    mean time: 0.404230 ms, 1.27 GB/s
flashinfer                               mean time: 0.178876 ms, speedup: 2.26, 2.86 GB/s
sampling_topk_topp_batched               mean time: 0.085515 ms, speedup: 4.73, 5.99 GB/s
####################################################################################################
bs: 1, vocab_size: 320000
torch                                    mean time: 0.689988 ms, 0.93 GB/s
flashinfer                               mean time: 0.168391 ms, speedup: 4.10, 3.80 GB/s
sampling_topk_topp_batched               mean time: 0.097644 ms, speedup: 7.07, 6.55 GB/s
####################################################################################################
bs: 8, vocab_size: 128000
torch                                    mean time: 0.745937 ms, 2.75 GB/s
flashinfer                               mean time: 0.107631 ms, speedup: 6.93, 19.03 GB/s
sampling_topk_topp_batched               mean time: 0.056521 ms, speedup: 13.20, 36.23 GB/s
####################################################################################################
bs: 8, vocab_size: 256000
torch                                    mean time: 1.306799 ms, 3.13 GB/s
flashinfer                               mean time: 0.222628 ms, speedup: 5.87, 18.40 GB/s
sampling_topk_topp_batched               mean time: 0.083780 ms, speedup: 15.60, 48.89 GB/s
####################################################################################################
bs: 8, vocab_size: 320000
torch                                    mean time: 1.586391 ms, 3.23 GB/s
flashinfer                               mean time: 0.239832 ms, speedup: 6.61, 21.35 GB/s
sampling_topk_topp_batched               mean time: 0.121928 ms, speedup: 13.01, 41.99 GB/s
####################################################################################################
bs: 16, vocab_size: 128000
torch                                    mean time: 1.058760 ms, 3.87 GB/s
flashinfer                               mean time: 0.198690 ms, speedup: 5.33, 20.62 GB/s
sampling_topk_topp_batched               mean time: 0.054836 ms, speedup: 19.31, 74.69 GB/s
####################################################################################################
bs: 16, vocab_size: 256000
torch                                    mean time: 2.176126 ms, 3.76 GB/s
flashinfer                               mean time: 0.288709 ms, speedup: 7.54, 28.37 GB/s
sampling_topk_topp_batched               mean time: 0.084949 ms, speedup: 25.62, 96.43 GB/s
####################################################################################################
bs: 16, vocab_size: 320000
torch                                    mean time: 2.873503 ms, 3.56 GB/s
flashinfer                               mean time: 0.390916 ms, speedup: 7.35, 26.19 GB/s
sampling_topk_topp_batched               mean time: 0.125600 ms, speedup: 22.88, 81.53 GB/s
####################################################################################################
bs: 32, vocab_size: 128000
torch                                    mean time: 2.185001 ms, 3.75 GB/s
flashinfer                               mean time: 0.272248 ms, speedup: 8.03, 30.09 GB/s
sampling_topk_topp_batched               mean time: 0.102874 ms, speedup: 21.24, 79.63 GB/s
####################################################################################################
bs: 32, vocab_size: 256000
torch                                    mean time: 4.398239 ms, 3.73 GB/s
flashinfer                               mean time: 0.611579 ms, speedup: 7.19, 26.79 GB/s
sampling_topk_topp_batched               mean time: 0.118800 ms, speedup: 37.02, 137.91 GB/s
####################################################################################################
bs: 32, vocab_size: 320000
torch                                    mean time: 5.618462 ms, 3.65 GB/s
flashinfer                               mean time: 0.812756 ms, speedup: 6.91, 25.20 GB/s
sampling_topk_topp_batched               mean time: 0.133236 ms, speedup: 42.17, 153.71 GB/s
```
