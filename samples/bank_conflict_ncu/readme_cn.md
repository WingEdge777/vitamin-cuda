# bank conflict between warps test

## 说明

load kernel test

- [x] load_fp32x4 kernel
- [x] pytorch op bindings && diff check

本代码用来做一个特殊情况的 bank conflict 验证的，搬运数据随便加个计算然后过一下 smem 立马写回。

网上大量的解释 bank conflict 概念和解决办法的文章，但是少有提到这种特殊情况的。那就是多个 warp 间同时访问 L1/smem 也会产生 bank conflict。

现代 gpu，一个 SM 都有多个 sub-core 调度器(大多4个)，多个 warp 是会同时向 L1/TEX/Smem 发起请求。当这些 warp 同时访问 smem 的相同 bank 的不同地址时，就会产生 bank conflict。。目前没有看到有解决这种冲突的办法，底层物理调度无可避免，只能降低冲突概率（比如向量化访问或者使用ldmatrix等降低请求分散度），再就是通过流水线计算时延隐藏来掩盖这种冲突开销。（现在异步拷贝+计算重叠已经是 kernel 优化必备了）

kernel 代码很简单，向量化写入/读取 smem，每个 warp 有 4 个内存事务（wavefronts），理论上没有 bank conflict。但是通过跑三个不同规模的数据实验就能看到，前两个小规模的是 0 冲突，最后一个大规模的就有了冲突。

具体看下面的第三个测试结果，266148 wavefronts 有 4004 次冲突（1.5%），嘿嘿~

## 测试

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
bash run_ncu.sh
```

### 输出

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

更多详细有关 bank conflict 理解和分析，不要看乱七八糟的博客了，可以直接参考 NV 技术报告：<https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41723/>
