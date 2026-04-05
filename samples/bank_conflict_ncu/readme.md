# Inter-Warp Bank Conflict Test

## Overview

A micro-benchmark to verify a subtle case of shared memory bank conflicts: **inter-warp conflicts**.

- [x] `load_fp32x4` kernel
- [x] PyTorch op binding & correctness check

This code demonstrates a rarely discussed scenario. The kernel simply moves data through SMEM with a trivial compute in between. Most articles on bank conflicts only cover intra-warp conflicts and solutions, but rarely mention that **multiple warps issuing requests simultaneously can also produce bank conflicts**.

Modern GPUs have multiple sub-core schedulers per SM (typically 4). Multiple warps can issue L1/TEX/SMEM requests concurrently. When these warps simultaneously access different addresses in the same SMEM bank, bank conflicts occur. There's no known way to eliminate this — it's an unavoidable physical scheduling artifact. The best we can do is reduce conflict probability (e.g., vectorized access or `ldmatrix` to reduce request scatter) and mask the conflict overhead with pipeline latency hiding (async copy + compute overlap is now essential for kernel optimization).

The kernel is straightforward: vectorized `float4` writes/reads to/from SMEM. Each warp generates 4 memory transactions (wavefronts), so there should be **zero intra-warp bank conflicts**. However, running three experiments at different scales reveals that the two small-scale tests show 0 conflicts, while the large-scale test shows conflicts.

Specifically, in the third test below: 266,148 wavefronts with 4,004 conflicts (~1.5%).

## Build & Test

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
bash run_ncu.sh
```

### NCU Output

```
   load_fp32x4_kernel(float *, float *, int) (1, 1, 1)x(128, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                        0
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                           16
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                           16
    smsp__inst_executed_op_shared_ld.sum                            inst            4
    smsp__inst_executed_op_shared_st.sum                            inst            4
    -------------------------------------------------------- ----------- ------------

  load_fp32x4_kernel(float *, float *, int) (128, 1, 1)x(128, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                        0
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                        0
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                         2048
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                         2048
    smsp__inst_executed_op_shared_ld.sum                            inst          512
    smsp__inst_executed_op_shared_st.sum                            inst          512
    -------------------------------------------------------- ----------- ------------

  load_fp32x4_kernel(float *, float *, int) (16384, 1, 1)x(128, 1, 1), Context 1, Stream 7, Device 0, CC 12.0
    Section: Command line profiler metrics
    -------------------------------------------------------- ----------- ------------
    Metric Name                                              Metric Unit Metric Value
    -------------------------------------------------------- ----------- ------------
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                     4004
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                        0
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                       266148
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                       262144
    smsp__inst_executed_op_shared_ld.sum                            inst        65536
    smsp__inst_executed_op_shared_st.sum                            inst        65536
    -------------------------------------------------------- ----------- ------------
```

For a thorough understanding of bank conflicts, refer to this NVIDIA GTC talk: <https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41723/>
