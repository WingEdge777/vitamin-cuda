# hgemm

## Overview

Half-precision GEMM kernels (BF16/FP16).

- [x] hgemm_cublas — bf16/fp16 (cuBLAS baseline)
- [x] hgemm_naive — bf16/fp16 (ldmatrix + MMA)
- [x] hgemm_bcf — bf16/fp16 (ldmatrix + MMA, As/Bs swizzle BCF, ~95–99% of cuBLAS)
- [x] hgemm_bcf_dbf — bf16/fp16 (ldmatrix + MMA, BCF swizzle, double buffer, faster than cuBLAS)
- [x] hgemm_bcf_dbf_rw — bf16/fp16 (ldmatrix + MMA, BCF, double buffer, coalesced global r/w, faster than cuBLAS)
- [x] pytorch op bindings && diff check

## Run tests

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

## Sample output

```yaml
####################################################################################################
n: 4096, m: 4096, k: 4096
torch                                    mean time: 4.097551 ms, 33.54 tflops
hgemm_cublas                             mean time: 4.210246 ms, speedup: 0.97, tflops: 32.64
hgemm_naive                              mean time: 5.191345 ms, speedup: 0.79, tflops: 26.47
hgemm_bcf                                mean time: 4.336920 ms, speedup: 0.94, tflops: 31.69
hgemm_bcf_dbf                            mean time: 4.096174 ms, speedup: 1.00, tflops: 33.55
hgemm_bcf_dbf_rw                         mean time: 4.075860 ms, speedup: 1.01, tflops: 33.72
```
