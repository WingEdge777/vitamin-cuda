<p align="center">
  <img src="posts/static/vw-logo.jpg" alt="Vitamin-CUDA" width="800">
</p>

<p align="center">
  <b>A hands-on CUDA dev learning path from novice to expert.</b>
</p>
<p align="center">
  <em>One Kernel a Day, Keeps High Latency Away. 🚀</em>
</p>

<!-- <p align="center">
  <a href="https://hellogithub.com/...">
    <img src="https://img.shields.io/badge/HelloGitHub-Review-blue" alt="HelloGitHub">
  </a>
</p> -->

<!-- <br> -->

Welcome to your daily dose of CUDA programming! **Vitamin-CUDA** is a curated collection of hands-on CUDA practices, designed to take you from **Hello World** to **High Performance**. Whether you are a beginner looking to understand the grid-stride loop or an enthusiast diving into warp-level primitives, there's a kernel here for you.

💻 Let's get started and happy coding! ⌨️

## News

- [2026.03.10] [sgemm_tf32](./kernels/sgemm/) tf32 `Tensor-Core` kernel outperforming cuBLAS `cp.async + double smem + swizzle + ldmatrix + mma`🚀(and stay tuned!)
- [2026.02.27] [sgemm](./kernels/sgemm/) SIMT kernel outperforming cuBLAS with `smem + swizzle + double buffer + coalesced r/w`🚀

## Contents 📖

- [Easy 🌟~🌟🌟](#easy)
- [Medium 🌟🌟~🌟🌟🌟](#medium)
- [Hard 🌟🌟🌟~🌟🌟🌟🌟](#hard)

## Prerequisites 🛠️

- NVIDIA GPU (Compute Capability 6.0+)
- CUDA Toolkit 11.0+
- C++ Compiler (GCC/Clang/MSVC)
- CMake 3.18+ (Optional, but recommended)
- PyTorch (For extension examples/python binding and performence comparation)

I recommend using nvidia pytorch ngc docker images for a quick start! Refer to <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch>

## Kernels (100+ kernels)

All kernels were tested on an RTX 5060 laptop (unless otherwise specified) and benchmarked against PyTorch 2.9

<div id="easy"></div>

### Easy (🌟~🌟🌟)

- [elementwise](./kernels/elementwise): elementwise add
  - [x] elementwise_add fp32/fp16 版
  - [x] elementwise_add_fp16x2(fp16向量化)
  - [x] elementwise_add_fp16x8(fp16向量化)
  - [x] elementwise_add_fp16x8(fp16向量化, packed r/w)
  - [x] pytorch op bindings && diff check
- [sigmoid](./kernels/sigmoid)
  - [x] sigmoid fp32/fp16 版
  - [x] sigmoid_fp16x2(fp16向量化)
  - [x] sigmoid_fp16x8(fp16向量化)
  - [x] sigmoid_fp16x8(fp16向量化, packed r/w)
  - [x] pytorch op bindings && diff check
- [swish](./kernels/swish)
  - [x] swish fp32/fp16 版
  - [x] swish_fp16x2(fp16向量化)
  - [x] swish_fp16x8(fp16向量化)
  - [x] swish_fp16x8(fp16向量化, packed r/w)
  - [x] pytorch op bindings && diff check
- [relu](./kernels/relu)
  - [x] relu fp32/fp16 版
  - [x] relu_fp16x2(fp16向量化)
  - [x] relu_fp16x8(fp16向量化)
  - [x] relu_fp16x8(fp16向量化, packed r/w)
  - [x] pytorch op bindings && diff check
- [relu6](./kernels/relu6)
  - [x] relu6 fp32/fp16 版
  - [x] relu6_fp16x2(fp16向量化)
  - [x] relu6_fp16x8(fp16向量化)
  - [x] relu6_fp16x8(fp16向量化, packed r/w)
  - [x] pytorch op bindings && diff check
- [elu](./kernels/elu)
  - [x] elu fp32/fp16 版
  - [x] elu_fp16x2(fp16向量化)
  - [x] elu_fp16x8(fp16向量化)
  - [x] elu_fp16x8(fp16向量化, packed r/w, half2 近两倍提升)
  - [x] pytorch op bindings && diff check
- [gelu](./kernels/gelu)
  - [x] gelu fp32/fp16 版
  - [x] gelu_fp16x2(fp16向量化)
  - [x] gelu_fp16x8(fp16向量化)
  - [x] gelu_fp16x8(fp16向量化，packed r/w)
  - [x] pytorch op bindings && diff check
- [hardswish](./kernels/hardswish)
  - [x] hardswish fp32/fp16 版
  - [x] hardswish_fp16x2(fp16向量化)
  - [x] hardswish_fp16x8(fp16向量化)
  - [x] hardswish_fp16x8(fp16向量化, packed r/w)
  - [x] pytorch op bindings && diff check
- [embedding](./kernels/embedding/)
  - [x] embedding fp32/fp16 版
  - [x] embedding_fp32x4(fp32向量化)
  - [x] embedding_fp32x4(fp32向量化, packed r/w)
  - [x] embedding_fp16x2(fp16向量化)
  - [x] embedding_fp16x8(fp16向量化)
  - [x] embedding_fp16x8(fp16向量化, packed r/w)
  - [x] pytorch op bindings && diff check
- [rope](./kernels/rope)
  - [x] pytorch naive rope
  - [x] pytorch rope with cos/sin table
  - [x] rope fp32 版 (比pytorch naive 实现快一个数量级)
  - [x] rope fp32x4 版 (fp32向量化，稍大规模后快几十倍)
  - [x] pytorch op bindings && diff check

<div id="medium"></div>

### Medium (🌟🌟~🌟🌟🌟)

- [reduce](./kernels/reduce/) : 基于 warp shuffle add
  - [x] reduce_sum fp32/fp16 版
  - [x] reduce_sum_fp16x2(fp16向量化)
  - [x] reduce_sum_fp16x8_packed(fp16向量化, packed r/w)
  - [x] reduce_sum int8 版
  - [x] reduce_sum_i8x16_packed (int8向量化，packed r/w)
  - [x] reduce_sum_i8x16_packed (int8向量化，packed r/w, dp4a, 相比torch朴素实现快几十倍)
  - [x] reduce_sum_i8x64_packed (int8向量化，packed r/w, dp4a)
  - [x] pytorch op bindings && diff check
- [dot_product](./kernels/dot_product)
  - [x] dot_product fp32/fp16 版
  - [x] dot_product_fp32x4(fp32向量化)
  - [x] dot_product_fp16x2(fp16向量化)
  - [x] dot_product_fp16x8(fp16向量化, packed r/w)
  - [x] pytorch op bindings && diff check
- [softmax](./kernels/softmax/)
  - [x] safe online softmax fp32/fp16 版
  - [x] safe online softmax fp32x4 版 (fp32向量化)
  - [x] safe online softmax fp16x8 版 (fp16向量化, packed r/w)
  - [x] pytorch op bindings && diff check
- [rmsnorm](./kernels/rmsnorm/)
  - [x] naive torch rmsnorm
  - [x] rmsnorm fp32/fp16 版
  - [x] rmsnorm fp32x4 版 (fp32向量化)
  - [x] rmsnorm_fp32x4_smem
  - [x] rmsnorm fp16x8 版 (fp16向量化, packed r/w)
  - [x] rmsnorm_fp16x8_smem 版 (fp16向量化, packed r/w)
  - [x] pytorch op bindings && diff check
- [transpose](./kernels/mat_transpose/)
  - [x] transpose_coalesced_read (input视角，合并读)
  - [x] transpose_coalesced_write (output视角，合并写)
  - [x] transpose_smem (共享内存缓存，块状读写)
  - [x] transpose_smem_bcf (共享内存无冲突版)
  - [x] transpose_smem_packed_bcf (共享内存无冲突版，float4向量化读写)
  - [x] transpose_smem_swizzled_packed (共享内存无冲突版，float4向量化读写)
  - [x] pytorch op bindings && diff check

<div id="hard"></div>

### Hard (🌟🌟🌟~🌟🌟🌟🌟)

- [sgemv](./kernels/sgemv/)
  - [x] gemv fp32版
  - [x] gemv fp32x4（向量化读取）
  - [x] pytorch op bindings && diff check
- [sgemm](./kernels/sgemm/)
  - [x] sgemm_cublas fp32 版
  - [x] sgemm_tiling (向量化读写 + block tiling共享内存版)
  - [x] sgemm_at_tiling (向量化读写 + a矩阵转置写入smem, 4-way 写入冲突, 内层循环float4读取)
  - [x] sgemm_at_bcf_swizzling (向量化读写 + at + swizzle， 无冲突版)
  - [x] sgemm_at_bcf_swizzling_rw (向量化读写 + at + swizzle + c写回事务合并)
  - [x] sgemm_at_bcf_swizzling_dbf_rw(向量化读写 + at + swizzle + c写回事务合并 + double buffer流水线, 超越cuBLAS ！)
  - [x] pytorch op bindings && diff check
- [sgemm_tf32](./kernels/sgemm/)
  - [x] sgemm_cublas tf32 版
  - [x] sgemm_tf32_bt (向量化读A/B，B转置写入smem, ldmatrix + mma)
  - [x] sgemm_tf32_bt_swizzle (向量化读A/B，B转置写入smem, ldmatrix + mma, As 0冲突)
  - [x] sgemm_tf32_bt_swizzle_dbf (向量化读A/B，B转置写入smem, ldmatrix + mma, As 0冲突, grid swizzling, 97~102% cuBLAS 性能)
  - [x] sgemm_tf32_swizzle_bcf (cp.async读写A/B，warp shuffle b寄存器转置， As/Bs无冲突, grid swizzling)
  - [x] sgemm_tf32_swizzle_bcf_dbf (cp.async读写A/B，warp shuffle b寄存器转置， As/Bs无冲突, grid swizzling，双buffer，超越cuBLAS)
  - [x] pytorch op bindings && diff check
- [hgemm](./kernels/hgemm/)
  - [x] hgemm_cublas bf16/fp16 版
  - [x] hgemm_naive bf16/fp16 版
  - [x] pytorch op bindings && diff check

<div id="samples"></div>

## Samples

- [deviceQuery](./samples/deviceQuery)
- [bankConflictNCU](./samples/bank_conflict_ncu/)

## Reference

- [LeetCUDA](https://github.com/xlite-dev/LeetCUDA)
- [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
