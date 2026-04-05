<p align="center">
  <img src="posts/static/vw-logo.jpg" alt="Vitamin-CUDA" width="800">
</p>

> 🍎 **One Kernel a Day, Keeps High Latency Away.**
>
> *A hands-on CUDA learning path from novice to expert.*

Welcome to your daily dose of GPU performance fuel. **Vitamin-CUDA** is a curated collection of 120+ hand-tuned CUDA kernels, ranging from basic grid-stride loops for beginners to cutting-edge asynchronous TMA pipelines for performance enthusiasts. Every kernel is seamlessly wrapped as a PyTorch C++ extension, delivering raw hardware performance with zero-overhead Python integration.

**No over-engineered abstractions. No black boxes. Just pure hardware saturation.**

## News

- **[2026.04.04]** [hgemm_120](./kernels/hgemm_sm120/) — FP16/BF16 Tensor-Core kernel pushing the absolute compute limit of the GPU. Outperforms cuBLAS via `TMA copy + 3-stage pipeline + double register buffer + swizzle + ldmatrix + mma`.
- **[2026.03.18]** [hgemm](./kernels/hgemm/) — FP16/BF16 Tensor-Core kernel outperforming cuBLAS via `cp.async + double SMEM buffer + swizzle + ldmatrix + mma`.
- **[2026.03.10]** [sgemm_tf32](./kernels/sgemm/) — TF32 Tensor-Core kernel outperforming cuBLAS via `cp.async + double SMEM buffer + swizzle + ldmatrix + mma`.
- **[2026.02.27]** [sgemm](./kernels/sgemm/) — SIMT kernel outperforming cuBLAS via `SMEM + swizzle + double buffer + coalesced read/write`.

## Contents

- [News](#news)
- [Contents](#contents)
- [Prerequisites](#prerequisites)
- [Kernels (120+)](#kernels-120)
  - [Easy](#easy)
  - [Medium](#medium)
  - [Hard](#hard)
- [Samples](#samples)
- [References](#references)

## Prerequisites

- NVIDIA GPU (Compute Capability 6.0+)
- CUDA Toolkit 11.0+
- C++ Compiler (GCC / Clang / MSVC)
- CMake 3.18+ (optional but recommended)
- PyTorch (for Python bindings and performance comparison)

> **Quick Start** — We recommend using the [NVIDIA PyTorch NGC Docker image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) for a ready-to-go environment.

## Kernels (120+)

All kernels were tested on an RTX 5060 Laptop GPU (unless otherwise noted) and benchmarked against PyTorch 2.9.

<div id="easy"></div>

### Easy

- [elementwise](./kernels/elementwise): element-wise add
  - [x] `elementwise_add` — FP32 / FP16
  - [x] `elementwise_add_fp16x2` — FP16 vectorized (×2)
  - [x] `elementwise_add_fp16x8` — FP16 vectorized (×8)
  - [x] `elementwise_add_fp16x8` — FP16 vectorized (×8, packed r/w)
  - [x] PyTorch op binding & correctness check
- [sigmoid](./kernels/sigmoid)
  - [x] `sigmoid` — FP32 / FP16
  - [x] `sigmoid_fp16x2` — FP16 vectorized (×2)
  - [x] `sigmoid_fp16x8` — FP16 vectorized (×8)
  - [x] `sigmoid_fp16x8` — FP16 vectorized (×8, packed r/w)
  - [x] PyTorch op binding & correctness check
- [swish](./kernels/swish)
  - [x] `swish` — FP32 / FP16
  - [x] `swish_fp16x2` — FP16 vectorized (×2)
  - [x] `swish_fp16x8` — FP16 vectorized (×8)
  - [x] `swish_fp16x8` — FP16 vectorized (×8, packed r/w)
  - [x] PyTorch op binding & correctness check
- [relu](./kernels/relu)
  - [x] `relu` — FP32 / FP16
  - [x] `relu_fp16x2` — FP16 vectorized (×2)
  - [x] `relu_fp16x8` — FP16 vectorized (×8)
  - [x] `relu_fp16x8` — FP16 vectorized (×8, packed r/w)
  - [x] PyTorch op binding & correctness check
- [relu6](./kernels/relu6)
  - [x] `relu6` — FP32 / FP16
  - [x] `relu6_fp16x2` — FP16 vectorized (×2)
  - [x] `relu6_fp16x8` — FP16 vectorized (×8)
  - [x] `relu6_fp16x8` — FP16 vectorized (×8, packed r/w)
  - [x] PyTorch op binding & correctness check
- [elu](./kernels/elu)
  - [x] `elu` — FP32 / FP16
  - [x] `elu_fp16x2` — FP16 vectorized (×2)
  - [x] `elu_fp16x8` — FP16 vectorized (×8)
  - [x] `elu_fp16x8` — FP16 vectorized (×8, packed r/w; ~2× speedup via `half2` intrinsics)
  - [x] PyTorch op binding & correctness check
- [gelu](./kernels/gelu)
  - [x] `gelu` — FP32 / FP16
  - [x] `gelu_fp16x2` — FP16 vectorized (×2)
  - [x] `gelu_fp16x8` — FP16 vectorized (×8)
  - [x] `gelu_fp16x8` — FP16 vectorized (×8, packed r/w)
  - [x] PyTorch op binding & correctness check
- [hardswish](./kernels/hardswish)
  - [x] `hardswish` — FP32 / FP16
  - [x] `hardswish_fp16x2` — FP16 vectorized (×2)
  - [x] `hardswish_fp16x8` — FP16 vectorized (×8)
  - [x] `hardswish_fp16x8` — FP16 vectorized (×8, packed r/w)
  - [x] PyTorch op binding & correctness check
- [embedding](./kernels/embedding/)
  - [x] `embedding` — FP32 / FP16
  - [x] `embedding_fp32x4` — FP32 vectorized (×4)
  - [x] `embedding_fp32x4` — FP32 vectorized (×4, packed r/w)
  - [x] `embedding_fp16x2` — FP16 vectorized (×2)
  - [x] `embedding_fp16x8` — FP16 vectorized (×8)
  - [x] `embedding_fp16x8` — FP16 vectorized (×8, packed r/w)
  - [x] PyTorch op binding & correctness check
- [rope](./kernels/rope) — Rotary Position Embedding
  - [x] PyTorch naive RoPE
  - [x] PyTorch RoPE with cos/sin look-up table
  - [x] `rope_fp32` — ~10× faster than the PyTorch naive implementation
  - [x] `rope_fp32x4` — FP32 vectorized; tens of times faster at larger sizes
  - [x] PyTorch op binding & correctness check

<div id="medium"></div>

### Medium

- [reduce](./kernels/reduce/) — Warp-shuffle-based reductions
  - [x] `reduce_sum` — FP32 / FP16
  - [x] `reduce_sum_fp16x2` — FP16 vectorized (×2)
  - [x] `reduce_sum_fp16x8_packed` — FP16 vectorized (×8, packed r/w)
  - [x] `reduce_sum` — INT8
  - [x] `reduce_sum_i8x16_packed` — INT8 vectorized (×16, packed r/w)
  - [x] `reduce_sum_i8x16_packed` — INT8 vectorized (×16, packed r/w, `dp4a`; tens of times faster than naive PyTorch)
  - [x] `reduce_sum_i8x64_packed` — INT8 vectorized (×64, packed r/w, `dp4a`)
  - [x] PyTorch op binding & correctness check
- [dot_product](./kernels/dot_product)
  - [x] `dot_product` — FP32 / FP16
  - [x] `dot_product_fp32x4` — FP32 vectorized (×4)
  - [x] `dot_product_fp16x2` — FP16 vectorized (×2)
  - [x] `dot_product_fp16x8` — FP16 vectorized (×8, packed r/w)
  - [x] PyTorch op binding & correctness check
- [softmax](./kernels/softmax/) — Safe online softmax
  - [x] One-pass
    - [x] Safe online softmax — FP32 / FP16
    - [x] Safe online softmax — FP32 vectorized (×4)
    - [x] Safe online softmax — FP16 vectorized (×8, pure register, packed r/w)
    - [x] Safe online softmax **medium** — FP16 vectorized (moderate register + SMEM, packed r/w)
    - [x] Safe online softmax **extreme** — FP16 vectorized (max register + SMEM, packed r/w)
  - [x] Two-pass (large rows beyond single-block capacity)
    - [x] Safe online softmax **arbitrary** — FP16 vectorized (max register + SMEM, packed r/w)
    - [x] Safe online softmax **split-k** — FP16 vectorized (max register + SMEM, packed r/w)
  - [x] PyTorch op binding & correctness check
- [rmsnorm](./kernels/rmsnorm/)
  - [x] PyTorch naive RMSNorm (baseline)
  - [x] `rmsnorm` — FP32 / FP16
  - [x] `rmsnorm_fp32x4` — FP32 vectorized (×4)
  - [x] `rmsnorm_fp32x4_smem` — with shared memory
  - [x] `rmsnorm_fp16x8` — FP16 vectorized (×8, packed r/w)
  - [x] `rmsnorm_fp16x8_smem` — FP16 vectorized (×8, packed r/w, shared memory)
  - [x] PyTorch op binding & correctness check
- [transpose](./kernels/mat_transpose/) — Matrix transpose
  - [x] `transpose_coalesced_read` — coalesced global loads (input-centric)
  - [x] `transpose_coalesced_write` — coalesced global stores (output-centric)
  - [x] `transpose_smem` — shared-memory tiled transpose
  - [x] `transpose_smem_bcf` — SMEM bank-conflict-free
  - [x] `transpose_smem_packed_bcf` — SMEM bank-conflict-free, `float4` vectorized r/w
  - [x] `transpose_smem_swizzled_packed` — SMEM swizzled, `float4` vectorized r/w
  - [x] PyTorch op binding & correctness check

<div id="hard"></div>

### Hard

- [sgemv](./kernels/sgemv/) — Single-precision GEMV
  - [x] `gemv_fp32` — baseline
  - [x] `gemv_fp32x4` — vectorized loads (×4)
  - [x] PyTorch op binding & correctness check
- [sgemm](./kernels/sgemm/) — Single-precision GEMM (SIMT)
  - [x] `sgemm_cublas` — cuBLAS FP32 baseline
  - [x] `sgemm_tiling` — vectorized r/w + block tiling with SMEM
  - [x] `sgemm_at_tiling` — A-transpose into SMEM (4-way bank conflict, inner-loop `float4` loads)
  - [x] `sgemm_at_bcf_swizzling` — A-transpose + swizzle, bank-conflict-free
  - [x] `sgemm_at_bcf_swizzling_rw` — + coalesced C write-back
  - [x] `sgemm_at_bcf_swizzling_dbf_rw` — + double buffer pipeline, **outperforms cuBLAS**
  - [x] PyTorch op binding & correctness check
- [sgemm_tf32](./kernels/sgemm/) — TF32 Tensor-Core GEMM
  - [x] `sgemm_cublas_tf32` — cuBLAS TF32 baseline
  - [x] `sgemm_tf32_bt` — vectorized loads for A/B, B-transpose into SMEM, `ldmatrix` + `mma`
  - [x] `sgemm_tf32_bt_swizzle` — + A-SMEM zero bank conflicts
  - [x] `sgemm_tf32_bt_swizzle_dbf` — + grid swizzling, 97–102% of cuBLAS
  - [x] `sgemm_tf32_swizzle_bcf` — `cp.async`, warp-shuffle B-register transpose, A/B SMEM conflict-free, grid swizzling
  - [x] `sgemm_tf32_swizzle_bcf_dbf` — + double buffer, **outperforms cuBLAS**
  - [x] PyTorch op binding & correctness check
- [hgemm](./kernels/hgemm/)
  - [x] `hgemm_cublas` — cuBLAS BF16/FP16 baseline
  - [x] `hgemm_naive` — BF16/FP16, `ldmatrix` + `mma`
  - [x] `hgemm_bcf` — BF16/FP16, `ldmatrix` + `mma`, A/B SMEM swizzle conflict-free, 95–99% of cuBLAS
  - [x] `hgemm_bcf_dbf` — + double buffer, **outperforms cuBLAS**
  - [x] `hgemm_bcf_dbf_rw` — + coalesced GMEM read/write, **outperforms cuBLAS**
  - [x] PyTorch op binding & correctness check
- [hgemm_sm120](./kernels/hgemm_sm120/)
  - [x] hgemm_cublas baseline
  - [x] hgemm_bcf_dbf_rw baseline: cp.async + ldmatrix + mma + double buffer + coalesced gmem r/w
  - [x] hgemm_k_stages: cp.async + ldmatrix + mma + 2/3 stages buffer + coalesced gmem
  - [x] hgemm_tma_r_k_stages_64: TMA read + ldmatrix + mma: one block, 3 stages, 128x128x64, double buffer register
  - [x] hgemm_tma_r_k_stages_32: TMA read + ldmatrix + mma: two blocks, 3 stages, 128x128x32, double buffer register
  - [x] pytorch op bindings && diff check

<div id="samples"></div>

## Samples

- [deviceQuery](./samples/deviceQuery)
- [bankConflictNCU](./samples/bank_conflict_ncu/)

## References

- [LeetCUDA](https://github.com/xlite-dev/LeetCUDA)
- [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
