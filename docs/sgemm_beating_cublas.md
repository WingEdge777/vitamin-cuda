# [CUDA 优化实战] sgemm - 超越 cuBLAS：带你学会极致优化的矩阵乘法 cuda c++ 实现

## 0. 序 - 标量时代的最后尊严

（究极多干货预警，配图较多，涉及大量位运算与内存映射推导代码，建议在 PC 端阅读以获得最佳体验）
（本文适用于有一定 CUDA 编程基础，知道基本矩阵乘法原理，以及对手撕 kernel 性能调优感兴趣的读者阅读）
（所有 kernel 完整代码可以从 github 获取：）
>
> 略有一点标题党，cuBLAS 毕竟是精确计算的通用库，是大量 NV 专家的心血集合，我当然没法完全打败 cuBLAS，但在特定情况下，不考虑边界检查等等，手搓算子超越 cuBLAS 是高性能计算工程师素养所在（做不到的，请坐小孩那桌，笑：))。
>
在如今 Tensor Core 满天飞的时代，写一个纯 FP32 的 SIMT 标量矩阵乘法（SGEMM）还有意义吗？有。因为它是检验一个底层计算工程师对 GPU 显存控制、Warp 调度、共享内存/寄存器资源分配以及指令级并行（ILP）理解的最强试金石。

本文以 M=N=K=4096（MxKxN,这是cuBLAS最擅长的中等体量规模）的矩阵乘法为例，在 RTX 5060 移动版显卡上，本人在不调用任何汇编级指令，不使用 cp.async 等新架构特性，仅用纯 C++ CUDA，将耗时从 PyTorch 的 16.59 ms 压榨到了 13.85 ms，在同精度赛道上成功超越了 NVIDIA 原厂的 cuBLAS（14.33 ms）。本文将复盘这场与 nvcc 编译器和物理硬件的完整较量细节。

本文会给出 6 个 kernel 实现，从基础的 tiling，共享内存的运用以及如何设计 swizzle 方案 解决 bank conflict，双 buffer 掩盖时延，尽可能的合并访问显存和 smem，提高指令级并行度，寄存器用量压制保障 Occupancy 等手段完成最终对 cuBLAS 的超越。整个过程能接触到具体到搬运/计算数据复杂精巧的下标映射，时延隐藏的哲学，强迫你进一步加深对自己的硬件了解，具体 kernel 大纲如下。话不多说，我们直接开始。

- sgemm_naive（无任何优化）
- sgemm_tiling （向量化读写 + block tiling 共享内存版）
- sgemm_at_tiling （向量化读写 + a 矩阵转置写入 smem, 4-way 写入冲突，内层循环 float4 读取）
- sgemm_at_bcf_swizzling （向量化读写 + at + swizzle， 无冲突版）
- sgemm_at_bcf_swizzling_rw （向量化读写 + at + swizzle + c 写回事务合并）
- sgemm_at_bcf_swizzling_dbf_rw（向量化读写 + at + swizzle + c 写回事务合并 + double buffer 流水线，超越 cuBLAS）

## 1. naive 实现

不多提，naive 的实现只是给看一眼矩阵乘法的最基础实现，就是把三重循环放到 GPU 上执行，每个线程负责 c 的一个元素，沿着 k 维度累加，不考虑任何优化。

```cpp
__global__ void sgemm_naive_kernel(float *a, float *b, float *c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.f;
    if (row < m && col < n) {
        for (int i = 0; i < k; i++) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

#define binding_func_gen(name, num, element_dtype)                                                                     \
    void name(torch::Tensor a, torch::Tensor b, torch::Tensor c) {                                                     \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        CHECK_T(c);                                                                                                    \
        const int M = a.size(0);                                                                                       \
        const int K = a.size(1);                                                                                       \
        const int N = b.size(1);                                                                                       \
        const dim3 threads_per_block(16, 16);                                                                          \
        const dim3 blocks_per_grid((N + 15) / 16, (M + 15) / 16);                                                      \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
                                                                                                                       \
        name##_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(                                              \
            a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);                                   \
    }
```

## 2. tiling + smem

过了这么多年，就不从太基础的版本开始，这里先给一个相对强的 2d data tiling + 2d block/thread tiling + shared memory 的 baseline 实现，然后开始演进。首先，我根据我的显卡性能（5060 移动版）和共享内存/寄存器资源大小（smem size 最大 100KB，每个 block 最多 48KB；每个 block 寄存器最多 65536 个），初步划定了 data tiling size 和线程 block size

![ab_tiling](static/tiling.svg)

- 以 c 矩阵为视角基础， data tiling BMxBN 为 128x128，thread block size 为 256
  - 一个 block 计算 c[128][128] 的子矩阵，256 个线程分为 8 个 warp(2x4)
    - 一个 warp 负责 c[64][32]，每个线程计算 8 行 8 列的 c[8][8] 的子矩阵
  - k 维度跨步 BK 为 16，一个 block 每次 load 128x16 个 a 和 16x128 个的 b
    - 16 是在 k 纬度上的切分，kernel 会有一个循环在 k 纬度上累加乘积和，最终得到 c[128][128] 的结果

有朋友要问了，你这是怎么定出来的数据分块和线程 block 大小。这里其实有很多因素考量，我可以谈几点。

关于线程 block：

- block 大小我肯定会从 128 和 256 里选，因为这是常规显卡设置 block 的甜点位。
- 其次这里有一个访存计算比的问题（下面会细说），那么我倾向选 256 线程，因为线程多了，可以 load 更多的数据到 smem，然后进行复用并计算 c 矩阵结果。

数据分块：

- 从 c 矩阵视角进行线程分配，线程数量定了 256 后，要 load/计算多少行列数据？
- 首先肯定是 4 的倍数，因为我要向量化访问。
- 我的显卡理论峰值 cuda core 算力为 3328*1455*1e6*2 ~= 9.6 TFLOPs，理论带宽峰值 12001*1e6*128/8*2 ~= 384GB/s，因此访存算力比为 9686/384 ~= 25.2 FLOPs/Byte
  - 为了越过内存墙，让 cuda core 疯狂发烧，在 load 完数据后执行 fma 次数要超过 25 次才好。最终，我挑了几种分块大小验证了下，最终选择对 c 进行 128x16x128 的数据分块。
    - 128*16*128*2 / (128*16*2) / 4 = 31.5 > 25.2

最后，其实个人直觉也占不小比重，我下意识就选了 128x128 的方阵分块。然后验证了下这个情况下的资源用量。假设 k 步长为 16。

- 共享内存用量：每个线程需要 load 128x16 的 a 和 16x128 的 b，共 128x16x2x4 bytes = 16KB 的显存数据，这个大小在 smem 容量范围内，且能被 256 整除，符合线程分配，可以有四个活跃 block。即使开 double buffer 32KB，也完全没问题，可以有 2 个活跃 block。
- 寄存器用量：每个线程需要 8x8 的 c 矩阵结果，共 64 个；其次搬运 a，b 数据需要 8*2=16 个寄存器，64+16 = 80，算上双 buffer 再加 16 个就是共 96 个寄存器，最后算上其他临时变量，中间结果，地址偏移变量等等估计 10~20+，256 线程使用总量肯定不超过 65536 个（65536/256 = 256），因此是也是安全的，而且理想情况至少有 2 个 block 活跃，如果超过 128 个寄存器用量就只能有一个 block 活跃了（事实上在初版 双 buffer 代码中我确实超过了 128 导致 Occupancy 降低而性能爆降，但通过移除了一些变量寄存器重新挽救了回来）。

如果把 BK 步长改为 8，访存计算比会降低,越不过内存墙；如果改为 32，那么 smem 和寄存器用量会暴增，Occupancy 下降，甚至没法双 buffer。

以上就是我选择 block size 和 data tiling size 的个人考量。当然了，有没有其他合适的参数呢？我觉得是有的，不过我筛出来的这份参数都满足我的需求，所以就直接用了。

这些参数在不同硬件上甚至不同数据规模下的选择完全可能是不同的，如果数据规模很小，那小一些的分块可能效果更好。在实际应用中，我们还可能会通过真实性能测试来 tune block 大小等等。

说了这么多，该上核心代码了：

```cpp
// a block calculate c[128][128], each thread c[8][8]
template <const int BM = 128, const int BN = 128, const int BK = 16, const int TM = 8, const int TN = 8>
__global__ void sgemm_tiling_kernel(float *a, float *b, float *c, int m, int n, int k) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x; // 0~255; 8 个 warp, 2x4 tiling;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // 一个 block 一次搬运 64x16 个 a， 8x128 个 b， 分两次搬运恰好共 128x16, 16x128
    // 每 4 个线程负责一行 a(16 个元素），每 32 个线程负责一行 b(128 个元素）
    int load_a_row = tid / 4;               // 0~63
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12...
    int load_b_row = tid / WARP_SIZE;       // 0~8
    int load_b_col = (tid % WARP_SIZE) * 4; // 0,4,8,12,16,20,24,28...

    // warp tiling, 每 4 个 warp 负责 c 的上下两部分 64x128，
    int warp_row = warp_id / 4;      // 0, 1
    int warp_col = warp_id % 4;      // 0, 1, 2, 3
    int t_row_in_warp = lane_id / 4; // 0~7
    int t_col_in_warp = lane_id % 4; // 0~3

    // c out 初始坐标， 每个线程负责 8 行 8 列 tile, 共 256 线程，256*64 = 128*128
    int c_row = warp_row * 64 + t_row_in_warp * 8;
    int c_col = warp_col * 32 + t_col_in_warp * 8;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float sum[TM][TN] = {0.f};

    // 沿 k 纬度循环
    for (int bk = 0; bk < k; bk += BK) {
        FLOAT4(As[load_a_row][load_a_col]) = FLOAT4(a[(by * BM + load_a_row) * k + bk + load_a_col]);
        FLOAT4(As[load_a_row + 64][load_a_col]) = FLOAT4(a[(by * BM + load_a_row + 64) * k + bk + load_a_col]);

        FLOAT4(Bs[load_b_row][load_b_col]) = FLOAT4(b[(bk + load_b_row) * n + bx * BN + load_b_col]);
        FLOAT4(Bs[load_b_row + 8][load_b_col]) = FLOAT4(b[(bk + load_b_row + 8) * n + bx * BN + load_b_col]);

        __syncthreads();

        // 8x8 循环计算累加乘积和，k 纬度
#pragma unroll
        for (int i = 0; i < BK; i++) {
            float reg_a[TM], reg_b[TN];

#pragma unroll
            for (int m_idx = 0; m_idx < TM; ++m_idx)
                reg_a[m_idx] = As[c_row + m_idx][i];

            FLOAT4(reg_b[0]) = FLOAT4(Bs[i][c_col]);
            FLOAT4(reg_b[4]) = FLOAT4(Bs[i][c_col + 4]);

#pragma unroll
            for (int m_idx = 0; m_idx < TM; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < TN; ++n_idx) {
                    sum[m_idx][n_idx] += reg_a[m_idx] * reg_b[n_idx];
                }
            }
        }
        __syncthreads();
    }

    // 写回 C 矩阵
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        FLOAT4(c[(by * BM + c_row + i) * n + bx * BN + c_col]) = FLOAT4(sum[i][0]);
        FLOAT4(c[(by * BM + c_row + i) * n + bx * BN + c_col + 4]) = FLOAT4(sum[i][4]);
    }
}

#define binding_tiled_func_gen(name)                                                                                   \
    void name(torch::Tensor a, torch::Tensor b, torch::Tensor c) {                                                     \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        CHECK_T(c);                                                                                                    \
        const int M = a.size(0);                                                                                       \
        const int K = a.size(1);                                                                                       \
        const int N = b.size(1);                                                                                       \
        const int BM = 128;                                                                                            \
        const int BN = 128;                                                                                            \
        const int threads_per_block = 256;                                                                             \
        const dim3 blocks_per_grid((N + BN - 1) / BN, (M + BM - 1) / BM);                                              \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
                                                                                                                       \
        name##_kernel<128, 128, 16, 8, 8><<<blocks_per_grid, threads_per_block, 0, stream>>>(                          \
            a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);                                   \
    }
```

注释里其实已经说得比较详细了，但这里我们再强调几个关键点：

- 搬运 a 数据块 128x16 和 b 数据块 16x128 就是粗暴的平铺线程，相邻线程跨列为 4，float4 向量化读取，相邻线程尽量读取连续地址合并访问
  - 每 4 个线程搬运 a 的一行，每 32 线程搬运 b 的一行
- 在分配计算 c 行列时，我先简单做了一个 2d warp tiling，每 4 个 warp 各负责上下两半 64x128 的 c，起初是觉得尽量均匀对称连续的分配行列比较好，然后每个线程各取连续的 8 行 a、8 列 b 进行累加乘积和
- 最后 c 写回，由于上一步取的是连续的列，所以也可以放心用 float4

## 3. at_tiling

初版的 kernel 有个明显的优化点，那就是在内层循环读取 As smem 时，用了循环标量读取，这给 LSU 带来了很大的指令压力，而且在计算循环中频繁的等待数据是很糟糕的选择，会产生很多小空泡。因此可以立马想到的一个优化点就是将 a 矩阵转置后存入 smem，这样读取时就能用 float4 等向量读取，提高指令级并行度。虽然这样导致写入时被迫以标量写入，但内层循环的读取压力大大降低，用外层循环 “两个 float4 写 变为 8 个 float 写入” 代替内层计算循环 “读 16x8 个 float 变为 16x 两个 float4” 这种 trade-off 是值得的，因为能显著降低 LSU 的压力，避免 ALU 计算单元在计算过程中等待 L1 数据，提高整体吞吐。

核心修改非常简单，就是交换一下 a 的行列，如下：

```cpp
template <const int BM = 128, const int BN = 128, const int BK = 16, const int TM = 8, const int TN = 8>
__global__ void sgemm_at_tiling_kernel(float *a, float *b, float *c, int m, int n, int k) {
    // 不变
    ...
    __shared__ float As_T[BK][BM]; // a 转置
    __shared__ float Bs[BK][BN];

    for (int bk = 0; bk < k; bk += BK) {
        // A 矩阵转置标量写入共享内存
        float4 tmp_a0 = FLOAT4(a[(by * BM + load_a_row) * k + bk + load_a_col]);
        As_T[load_a_col + 0][load_a_row] = tmp_a0.x;
        As_T[load_a_col + 1][load_a_row] = tmp_a0.y;
        As_T[load_a_col + 2][load_a_row] = tmp_a0.z;
        As_T[load_a_col + 3][load_a_row] = tmp_a0.w;

        float4 tmp_a1 = FLOAT4(a[(by * BM + load_a_row + 64) * k + bk + load_a_col]);
        As_T[load_a_col + 0][load_a_row + 64] = tmp_a1.x;
        As_T[load_a_col + 1][load_a_row + 64] = tmp_a1.y;
        As_T[load_a_col + 2][load_a_row + 64] = tmp_a1.z;
        As_T[load_a_col + 3][load_a_row + 64] = tmp_a1.w;

        ...

        // 8x8 循环计算累加乘积和
#pragma unroll
        for (int i = 0; i < BK; i++) {
            float reg_a[TM], reg_b[TN];
            // float4 读取
            FLOAT4(reg_a[0]) = FLOAT4(As_T[i][c_row]);
            FLOAT4(reg_a[4]) = FLOAT4(As_T[i][c_row + 4]);
            // 其他不变
            ...
        }
        ...
    }
    ...
}
```

## 4. at_tiling + swizzling

上一版代码虽然通过转置 As 矩阵提高了内层循环读取效率，但是也引入了一个新的问题：shared memory bank conflict。在标量写入 smem 时，发生了 4-way bank conflict。原因就在于

```cpp
    int load_a_row = tid / 4;               // 0~63
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12...
    ...
    As_T[load_a_col + 0][load_a_row] = tmp_a0.x; // 4-way bank conflict
```

为什么会冲突？首先直观看待一下 As_T[16][128]， 一行 128 个数据，所以行号不影响 bank id（bank id = (row*128 + col) % 32）, 而搬运 a 数据时为了合并访问线程是平铺的，导致在转置写入 smem 时，单个 warp 内相邻线程对应的坐标行列号变为了（以 warp 0 为例）

```yaml
col : 0, 0, 0, 0, 1, 1, 1, 1...
row : 0, 4, 8, 12, 0, 4, 8, 12...
```

典型的 4-way bank conflict，每四个线程同时访问同一 bank 的不同地址。这里我们使用一个稍微复杂一点的 swizzle 技巧来避免 bank conflict。我们需要找到一种映射 f(row, col) -> (row, new_col) 使得 new_col 分散在连续的 32 个 bank 内。

在之前矩阵转置的文章提到了一种简单 XOR swizzle 的方法：`new_col = row^col` 这里没法直接用，为啥？因为这里的 col 和 row 都没有遍历一组 32 个 bank 的所有可能值，直接使用 row 的 bit 去扰动 col，并不能达到目的。

好，我们现在开始推导新的 swizzle 公式。首先牢记一个物理限制：Bank ID 只由坐标的低 5 bits 决定（因为 % 32）。

![swizzling](static/swizzle.svg)

通过仔细观察，一个 warp 内，以 warp 0 为例：

- row（即代码中的 load_a_col）的值是 0，4，8，12，... ，其二进制表示的变量位在 bit 2 和 bit 3（00000, 00100, 01000, 01100...）。
- 而 col（即代码中的 load_a_row）取值是 0~7，其二进制的变量位在 bit0~bit2（00000 ~ 00111）。

很自然的，既然冲突是相同的 col 导致的，而他们的 row 值都不同。那我们可以把 row 左移一位，变成 0, 8, 16, 24，这样它的变量位就推到了 bit3 和 bit4。此时再把它和 col 做 XOR，神奇的事情就发生了：row 负责填满高 2 位（bit 3, 4），col 负责填满低 3 位（bit 0~2），两者交错异或，相当于完美铺满并遍历了低 5 bits 的所有 32 种情况！

其他 warp 情况是类似的，比如 warp 1 的 col 为 01xxx， 只要去异或 bit4~5 的四种排列，就能得到互不相同的 32 个值，这是 xor 的双射性质保证的；而单个 warp 内 row，每次都是同步变动，比如在 row 为 1，5，9，13 时， 二进制为 xx01，左移 1 位后变化位依然是 bit4~5，同理依然能得到互不相同的 32 个值

由此，我们得到一个写入 As_T 无冲突的 swizzle 映射：`new_col = col^(row << 1)`

但这还不够，虽然乱序写入完美打散了 Bank，但我们在内层循环还需要用 float4 向量化读取，float4 的物理底线是：这 4 个连续的 float 不仅要在物理内存上挨在一起，且起始地址必须 16 字节对齐。同时我们注意到，在`FLOAT4(reg_a[0]) = FLOAT4(As_T[i][SWIZZLE_A(i, c_row)]);`这个读取动作中，32 个线程的外层 row 维坐标（变量 i）是固定不变的！

怎么做？col 下标的二进制的低 2bits 决定了 4 float 的一个 segment，只需要把 row 的低 2bits 都抹为 0，`00 xor (col's bit0~1) = col‘s’ bit0~1`，就不会改变 `col+0,col+1,col+2,col+3` 4 个元素之间的相对顺序。这，其实就是 float4 数据打包的地址 swizzle 映射：`new_col = col^((row>>2) << 3)`。

这个映射确保了 row 的扰动有效位始终是 bit4~5 (>>2 把低 bit0~1 都抹掉了，当 i 遍历 0，1，2，3 时，(i>>2) << 3 值都为 0；当 i 遍历 4,5,6,7 时，(i>>2) << 3 值都为 8，以此类推）

最终核心修改：

```cpp
#define SWIZZLE_A(x, y) ((y) ^ ((x >> 2) << 3))

template <const int BM = 128, const int BN = 128, const int BK = 16, const int TM = 8, const int TN = 8>
__global__ void sgemm_at_bcf_swizzling_kernel(float *a, float *b, float *c, int m, int n, int k) {
    // 不变
    ...
    for (int bk = 0; bk < k; bk += BK) {
        // A 矩阵转置并 swizzling 写入共享内存
        float4 tmp_a0 = FLOAT4(a[(by * BM + load_a_row) * k + bk + load_a_col]);
        As_T[load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row)] = tmp_a0.x;
        As_T[load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row)] = tmp_a0.y;
        As_T[load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row)] = tmp_a0.z;
        As_T[load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row)] = tmp_a0.w;

        float4 tmp_a1 = FLOAT4(a[(by * BM + load_a_row + 64) * k + bk + load_a_col]);
        As_T[load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row + 64)] = tmp_a1.x;
        As_T[load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row + 64)] = tmp_a1.y;
        As_T[load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row + 64)] = tmp_a1.z;
        As_T[load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row + 64)] = tmp_a1.w;

        ...

        // 8x8 循环计算累加乘积和
#pragma unroll
        for (int i = 0; i < BK; i++) {
            // swizzle 读取
            FLOAT4(reg_a[0]) = FLOAT4(As_T[i][SWIZZLE_A(i, c_row)]);
            FLOAT4(reg_a[4]) = FLOAT4(As_T[i][SWIZZLE_A(i, c_row + 4)]);

            ...
        }
        __syncthreads();
    }
}
```

全面的 swizzle 机制其实可以单开一篇文章详细写写。但在此要重点说明一下，swizzle 的具体公式是要依据数据类型和 data layout 推导出来的（cutlass 的 swizzle 模板也只能用于几种特定的 data + layout）。只要理解 xor 的性质，推导 xor swizzle 公式并不难，使用起 swizzle 模板也更得心应手。

## 5. at_tiling + swizzling + 全合并读写 global memory

上一版作为单 buffer 无流水线的 kernel 已经比较可以了，但还有两点瑕疵，使用 ncu profile 后查看发现有

- 非合并写 global memory 的访存写入，提示如下

```yaml
1、
The memory access pattern for global stores to L1TEX might not be optimal. On average, only 16.0 of the 32 bytes transmitted per sector are utilized by each thread. This could possibly be caused by a stride between threads. Check the  Source Counters section for uncoalesced global stores.

2、
This kernel has uncoalesced global accesses resulting in a total of 2097152 excessive sectors (2% of the total 138412032 sectors). Check the L2 Theoretical Sectors Global Excessive table for the primary source locations. The  CUDA Programming Guide has additional information on reducing uncoalesced device memory accesses.
```

为什么 ncu 会出现这两个提示，原因是在单个线程计算 8 行 8 列时，我们选择了连续的 8 列，这导致在写回 c 矩阵 时需要分两次 float4 写出，如果看一个 warp 相邻线程的情况就是，都隔着空泡(4个 float) 在写数据，而 L1/L2 的访问模式都是按一个 128bytes 作为内存事务进行的，我们分两次写出，其实都发起了两批完全一模一样的内存事务请求，只不过第一次写了其中 64bytes，第二次写了另外 64bytes，这造成了显著浪费。

两个异常提示是同一个问题在L1和L2两个层面体现，为了解决这个问题。我们可以重新设计一下每个线程负责的 8 列，只要把这 8 列分成两个连续的 4 列，依然可以用 float4 读写，并且可以把相邻线程划到连续的 float4 地址上，比如 T0 读取 0~3，T1 读取 4~7..., 读完一次 float4 后， T0 再读 64~67，T1 读取 68~71... 这样就能保证写回 c 时相邻线程地址是合并的了。

![col_shuffle](static/col_shuffle.svg)

核心修改如下，把 2d warp tiling 去掉，一个 warp 负责 16 行 c，然后直接用线程 id 去映射成我们想要的 c_col。

```cpp
    // warp tiling
    // 线程在 Warp 内的行偏移依然是 0 或 8
    int t_row_in_warp = (lane_id / 16) * 8;

    // 每个 Warp 只负责 16 行，每 16 线程负责 8 行 128 列，每行 128 个元素，列维度分两次 load + 计算
    // 每个线程 一次 load 8 行 8 列，但是 8 列拆开为跨越 64 列的两次 float4, 比如 T0 负责读写 0~3,64~67 8 列，
    // 这样每 8 个线程在写回 c 的时候是连续的 32 个 float,128bytes, 完美事务合并
    int c_row = warp_id * 16 + t_row_in_warp;
    int c_col_base = (lane_id % 16) * 4;
    int c_col_0 = c_col_base;      // 0~3
    int c_col_1 = c_col_base + 64; // 64~67
```

## 6. at_tiling + swizzling + 全合并读写 global memory + 双 smem buffer

上一个 kernel 写完，其实基本达到单 buffer kernel 的天花板了，完美的合并读写（global memory，smem）,充分的访存算力比。还想进一步提高上限，要开始上特殊技巧了，那就是 copy 和 compute overlap，这是高性能计算老生常谈的话题了。CUDA LSU 发起内存事务请求，从 global memory 加载数据到寄存器（其实是要过 L2-->L1-->register）, ALU 可以说是没事干，即使有多个 block/warp 在切换计算，但还是不如在一个 warp 内利用空闲时间做计算来得高效（进一步提高指令级并行度，隐藏访问时延）。这里实现的方式是传统的 double buffering, 不使用 cp.async 等新架构特性。

流水线算法流程：

- 预先加载一块 a/b 到 smem_buffer[0]，同步`__syncthreads()`确保写入smem完成
- 循环主体：
  - 先发起请求加载下一块 a/b 到 寄存器，
  - 然后立刻拿 smem_buffer[0] 中的数据计算（和上一步overlap）
  - 计算完后，再将循环开头加载到寄存器的数据，写入到 smem_buffer[1]，同步`__syncthreads()`
    - 对比单buffer kernel，可以发现，循环主体少了一次同步开销（其实就是空间换时间）
  - 交换 smem_buffer 指针，开始下一循环
- 收尾阶段
  - 循环结束，计算最后一次 load 的数据块
  - 将累加寄存器 sum[8][8] 写回 c 矩阵的global memory，算法结束

最后贴一个最终版本的完整 kernel 代码：

```cpp

template <const int BM = 128, const int BN = 128, const int BK = 16, const int TM = 8, const int TN = 8>
__global__ void sgemm_at_bcf_swizzling_dbf_rw_kernel(float *a, float *b, float *c, int m, int n, int k) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x; // 0~255; 8 个 warp
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // 搬运映射
    int load_a_row = tid / 4;               // 0~63
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12...
    int load_b_row = tid / WARP_SIZE;       // 0~8
    int load_b_col = (tid % WARP_SIZE) * 4; // 0,4,8,12,16,20,24,28...

    // c 计算读写数据映射，和之前相同
    int t_row_in_warp = (lane_id / 16) * 8;
    int c_row = warp_id * 16 + t_row_in_warp;
    int c_col_base = (lane_id % 16) * 4;
    int c_col_0 = c_col_base; // 0~3
    // int c_col_1 = c_col_base + 64; // 64~67 减少变量，压低寄存器用量

    // double buffer
    __shared__ float As_T[2][BK][BM];
    __shared__ float Bs[2][BK][BN];

    float sum[TM][TN] = {0.f};

    // 维护显存读取的一维扁平指针，方便在流水线中步进
    float *a_ptr = a + (by * BM + load_a_row) * k + load_a_col;
    // float *a_ptr_64 = a + (by * BM + load_a_row + 64) * k + load_a_col;
    float *b_ptr = b + load_b_row * n + bx * BN + load_b_col;
    // float *b_ptr_8 = b + (load_b_row + 8) * n + bx * BN + load_b_col;

    // 先加载第一块，多消耗 16 个寄存器，上面几个注释掉将寄存器数量压低于 128 个，保障 Occupancy 不变
    float4 tmp_a0 = FLOAT4(a_ptr[0]);
    float4 tmp_a1 = FLOAT4(a_ptr[64 * k]);
    float4 tmp_b0 = FLOAT4(b_ptr[0]);
    float4 tmp_b1 = FLOAT4(b_ptr[8 * n]);

    As_T[0][load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row)] = tmp_a0.x;
    As_T[0][load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row)] = tmp_a0.y;
    As_T[0][load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row)] = tmp_a0.z;
    As_T[0][load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row)] = tmp_a0.w;

    As_T[0][load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row + 64)] = tmp_a1.x;
    As_T[0][load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row + 64)] = tmp_a1.y;
    As_T[0][load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row + 64)] = tmp_a1.z;
    As_T[0][load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row + 64)] = tmp_a1.w;

    FLOAT4(Bs[0][load_b_row][load_b_col]) = tmp_b0;
    FLOAT4(Bs[0][load_b_row + 8][load_b_col]) = tmp_b1;

    __syncthreads();

    // double buffer 下标
    int write_idx = 1;
    int read_idx = 0;
    // 主循环
    for (int bk = BK; bk < k; bk += BK) {
        // 沿 k 纬度偏移指针
        a_ptr += BK;
        b_ptr += BK * n;

        // 加载下一批数据，这个是异步的，发射完 ldg 指令后，可以立刻开始计算之前读取的数据
        tmp_a0 = FLOAT4(a_ptr[0]);
        tmp_a1 = FLOAT4(a_ptr[64 * k]);
        tmp_b0 = FLOAT4(b_ptr[0]);
        tmp_b1 = FLOAT4(b_ptr[8 * n]);

        // 计算逻辑和之前完全相同
#pragma unroll
        for (int i = 0; i < BK; i++) {
            float reg_a[TM], reg_b[TN];

            FLOAT4(reg_a[0]) = FLOAT4(As_T[read_idx][i][SWIZZLE_A(i, c_row)]);
            FLOAT4(reg_a[4]) = FLOAT4(As_T[read_idx][i][SWIZZLE_A(i, c_row + 4)]);

            FLOAT4(reg_b[0]) = FLOAT4(Bs[read_idx][i][c_col_0]);
            FLOAT4(reg_b[4]) = FLOAT4(Bs[read_idx][i][c_col_0 + 64]);

#pragma unroll
            for (int m_idx = 0; m_idx < TM; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < TN; ++n_idx) {
                    sum[m_idx][n_idx] += reg_a[m_idx] * reg_b[n_idx];
                }
            }
        }

        // 计算完，把上面异步加载的寄存器数据写入共享内存
        As_T[write_idx][load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row)] = tmp_a0.x;
        As_T[write_idx][load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row)] = tmp_a0.y;
        As_T[write_idx][load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row)] = tmp_a0.z;
        As_T[write_idx][load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row)] = tmp_a0.w;

        As_T[write_idx][load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row + 64)] = tmp_a1.x;
        As_T[write_idx][load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row + 64)] = tmp_a1.y;
        As_T[write_idx][load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row + 64)] = tmp_a1.z;
        As_T[write_idx][load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row + 64)] = tmp_a1.w;

        FLOAT4(Bs[write_idx][load_b_row][load_b_col]) = tmp_b0;
        FLOAT4(Bs[write_idx][load_b_row + 8][load_b_col]) = tmp_b1;

        __syncthreads(); // 同步，然后开始下一次循环
        write_idx ^= 1;
        read_idx ^= 1;
    }
    // 最后还有一批数据要计算
#pragma unroll
    for (int i = 0; i < BK; i++) {
        float reg_a[TM], reg_b[TN];

        FLOAT4(reg_a[0]) = FLOAT4(As_T[read_idx][i][SWIZZLE_A(i, c_row)]);
        FLOAT4(reg_a[4]) = FLOAT4(As_T[read_idx][i][SWIZZLE_A(i, c_row + 4)]);

        FLOAT4(reg_b[0]) = FLOAT4(Bs[read_idx][i][c_col_0]);
        FLOAT4(reg_b[4]) = FLOAT4(Bs[read_idx][i][c_col_0 + 64]);

#pragma unroll
        for (int m_idx = 0; m_idx < TM; ++m_idx) {
#pragma unroll
            for (int n_idx = 0; n_idx < TN; ++n_idx) {
                sum[m_idx][n_idx] += reg_a[m_idx] * reg_b[n_idx];
            }
        }
    }
    // pipeline 完成，写回 c
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        FLOAT4(c[(by * BM + c_row + i) * n + bx * BN + c_col_0]) = FLOAT4(sum[i][0]);
        FLOAT4(c[(by * BM + c_row + i) * n + bx * BN + c_col_0 + 64]) = FLOAT4(sum[i][4]);
    }
}
```

细心的读者可能会发现，我在最终代码里注释掉了一些中间变量（比如 c_col_1, a_ptr_64），转而在访问时直接计算偏移量 a_ptr[64 * k]。这是因为 Double Buffering 会让读写寄存器用量倍增，如果不精打细算，很容易超过每个线程的寄存器阈值限制（我的初始代码单线程就超过128个寄存器，导致活跃 warp 减半性能暴跌）。通过移除这些非必要的中间变量，我成功把寄存器用量压回到了128内，保住了 Occupancy。

## 7. benchmark 结果和分析

选择 M=N=K=4096 并非随意之举。首先，4096 作为完美的 2 的幂次方，能让 cuBLAS 毫无边界判断包袱地跑在 Fast Path 上，展现其最强实力；其次，4096 也是当前主流 7B/8B 大语言模型的标准隐层维度，在 LLM 的 Prefill 阶段极为常见。在这个‘cuBLAS 的绝对主场’也是‘现代 AI 最核心的计算场景’中正面硬刚，更能检验出我们手搓 Kernel 的含金量。

show code 部分结束，说理论性能多好没意义，口说无凭，直接上 benchmark 结果和 ncu profile报告

```bash
####################################################################################################
n: 4096, m: 4096, k: 4096
torch                          mean time: 14.974799 ms
sgemm_cublas                   mean time: 14.523163 ms, speedup: 1.03
sgemm_tiling                   mean time: 18.760985 ms, speedup: 0.80
sgemm_at_tiling                mean time: 16.436968 ms, speedup: 0.91
sgemm_at_bcf_swizzling         mean time: 15.706529 ms, speedup: 0.95
sgemm_at_bcf_swizzling_rw      mean time: 15.522802 ms, speedup: 0.96
sgemm_at_bcf_swizzling_dbf_rw  mean time: 14.193397 ms, speedup: 1.06
sgemm_cublas_tf32              mean time:  8.798057 ms, speedup: 1.70
```

从 18.760985 到 14.193397，纯手搓的性能优化，正面硬刚超越 cuBLAS ！这整个过程，我们通过精确地资源分配，设计复杂而精巧的搬运/计算坐标映射，加上双 buffer 流水线技术，实现了超越 cuBLAS 的性能。相信熟练并掌握这些技巧之后，面对绝大多数矩阵相关的算法 kernel 实现，都有信心轻松应对。

### ncu 报告

![ncu report](static/sgemm_ncu.png)

从报告可以看到，ncu 已经被 `sgemm_at_bcf_swizzling_dbf_rw kernel` 完全打服了，`estimated speedup = 0%`，说明 ncu 认为该 kernel 已经达到了理论性能上限。反而第一个 cuBLAS 的 kernel 还有“提高空间”（笑：）

## 8. 一些讨论

通过 ncu 的 report 我们还可以看到一些有意思的东西

- 比如我的前两个 swizzling kernel 依然提示了 bank conflict，但 double buffering 之后就没提示，尽管我的 swizzling 逻辑是完全一样的。为什么呢？这种 bank conflict 的出现是由于向量化读写导致的，一个 warp 使用 float4 读写 128个float，512bytes，无论怎么写都要 4 个内存事务请求完成，但是一条指令在一个周期内没有完成就会被认为是 bank conflict。实际对性能没影响，而且 float4 读写是更高效的（减少了指令开销）。既然物理限制无法打破，那我们就用架构设计来掩盖它！Double Buffering 流水线的作用，就在于它能把这种底层无法消灭的硬件冲突延迟，完美隐藏在庞大的计算流之下，让 NCU 都认为它不再是瓶颈。
- report 中还能看到 cuBLAS 其实是调用了 cutlass 的 kernel（void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(T1::Params)），cutlass SIMT 算子命名规则是`[M]x[N]_[K]x[Stages]`
  - 因此 cuBLAS 对 c 矩阵 tiling 的选择是 128x64（m=128，n=64），跨步 k=8，并且用了恐怖的 5 级流水线来隐藏时延。
  - block size 设置为 128（对应较小的数据搬运量），grid 用了奇怪的 512x4（大概率用了 grid swizzling，重新排布了 block 访问显存的顺序以吃到 L2cache 红利）
  - 如果点开 cuBLAS kernel 的 `memory workload analysis`，可以看到有巨多的 `shared store from global load bank conflict`，说明了啥，cuBLAS 使用了 cp.async 指令来异步加载数据，而它在这里也容忍了 Bank Conflict，是因为硬件层面异步拷贝指令（cp.async）的位宽和事务打包机制本身就与传统 ldg 结合共享内存写入不同。一方面无可避免，另一方面也说明了只要流水线深度足够，这种级别的冲突时延往往被完全掩盖，进一步印证了 Double Buffering 的正确性。
- Tensor Core的无敌。我们纯手搓的 SIMT Kernel 跑出了 14.19 ms，而 cuBLAS TF32 跑出了 8.79 ms。这说明在利用了 Tensor Core 的情况下，算力天花板被拉高了近一倍。

## 总结

最后，在 Tensor Core 统御大模型计算的今天，深挖 SIMT 编程模型的极限，不只是为了打败 cuBLAS 的那一两毫秒，更是为了保持对底层硬件心智模型的深度理解，有了这种理解与掌控，相信使用新架构新特性也能更加从容。

如有错误，请大家指正。欢迎大家来交流学习！

以上。
