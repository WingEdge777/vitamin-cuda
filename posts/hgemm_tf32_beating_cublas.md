# [CUDA 优化实战] hgemm - 超越 cuBLAS：Tensor-core、cp.async、ldmatrix、mma

## 0. 序 - 半精度一统江湖
>
> 本文适用于有一定 CUDA 编程基础，熟悉 GEMM 优化，对进阶 tensor core / 嵌入 PTX 指令 性能调优感兴趣的读者阅读
>
> 所有 kernel 完整代码可以从 github 获取，欢迎大家关注我的手撕算子系列 vitamin-cuda 项目：
>
> 没错，这是超越 cuBLAS 系列之三（NV 还不针对移动端显卡调优的话，我们就还是超越。不过这估计是 GEMM 系列最后一篇了，也可能会写 fp8 的 kernel 但应该不会是纯血 fp8 gemm 了，感觉有点同质化。而且 fp8 本身就要搭配各种量化姿势使用）
>

现在 bf16 甚至 fp8 几乎统治了 llm/vlm 训练和推理，传统模型可能还有在用 fp16（精度略高一些，但动态范围小），传统模型的权重和激活值分布比较集中，离群值相对少，用 fp16 更佳。而 LLm、VLM 现在一般原始模型是 bf16，推理可能会使用 fp8/int8/int4 等各种量化姿势。所以半精度矩阵乘法才是真正的主战场。

本文以 M=N=K=4096（MxKxN, cuBLAS 最擅长的中等规模）的 GEMM fp16/bf16 为例，在 RTX 5060 移动版显卡上，使用 cp.async、ldmatrix、mma 等 PTX 指令，配合 Tensor Core 加速计算，在同精度赛道上成功超越了 NVIDIA 原厂的 cuBLAS。本文将复盘这场与 cuBLAS 较量的过程。介绍 cp.async 指令、ldmatrix/mma warp 级别 PTX 指令的运用，硬核的 swizzle 推导，layout 分析与理解等等

本文会给出 4 个 kernel 实现，从基础的 grid swizzle + cp.async + 双 ldmatrix + mma，到 swizzlling 读写 smem 解决 bank conflict，上 double buffer 隐藏时延，最后合并 gmem 读写事务（是的，填了上一篇文章的 c 矩阵写回 gmem 事务合并的坑） 等手段完成最终对 cuBLAS 的超越。整个过程涉及对指令要求/用法的介绍和分析，swizzle 设计，希望能帮助读者深入理解 Tensor Core 的使用和优化技巧。

kernel 大纲如下（第一个是 cuBLAS kernel）

- hgemm_cublas bf16/fp16 版
- hgemm_naive bf16/fp16 版 (ldmatrix + mma)
- hgemm_bcf bf16/fp16 版 (ldmatrix + mma, As/Bs swizzle bcf, 95~99% cuBLAS' performance)
- hgemm_bcf_dbf bf16/fp16 版 (ldmatrix + mma, As/Bs swizzle bcf, double buffer, outperforming cuBLAS)
- hgemm_bcf_dbf_rw bf16/fp16 版 (ldmatrix + mma, As/Bs swizzle bcf, double buffer, coalesced r/w gmem, outperforming cuBLAS)

## 1. hgemm_naive

我们之前的文章已经介绍过 cp.async, ldmatrix, mma 的基本用法，这里再简单提一下，只说用到指令的具体用法，省略输入/输出列表，详细用法请参考官方文档

```cpp
cp.async.cg.shared.global.L2::128B [%0], [%1], 16; // cg： (Cache at Global level)：bypass L1，拷贝 16 字节，prefetch 128B 到 L2
cp.async.commit_group;  // 提交异步拷贝任务
cp.async.wait_group 0;  // 表示允许当前线程后台异步的 group 数，0 表示不允许后台，要等待到全部完成
```

- `cp.async`：使用姿势和 tf32 的 kernel 中的一样，毫无变化

```cpp
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32        d, a, b, c;
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32        d, a, b, c;
```

- `mma`指令：有一点变化
  - 之前计算 shape 是 m16n8k8（简称 1688），现在是 m16n8k16
  - 之前用 a/b 类型使用 tf32，现在换成 f16/bf16

```cpp
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];
ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];
```

- ldmatrix：一个 warp 从 smem 协同加载一个小矩阵分块到所有线程，所有线程一起 hold 着的寄存器结果叫做一个 fragment（输出所以线程 hold 着的值叫 fragment c）
  - 注意：x2，x4，表示读取线程数为 16，32。ldmatrix 读取数据，提供地址的每个线程永远是读 16 字节！读取完后会分发到各个线程，组成一个 fragment。
    - 理解这一点，才能理解如何解决 bank conflict
  - 由于半精度计算我们使用 mma 的 m16n8k16 shape，所以要 从 As 读取 16x16 的 tile，和 Bs 的 16x8 的 tile

本次我们有 tf32 的经验，所以再简单说下 tiling 策略，就直接上代码。

- 128x128 的 tiling（c 矩阵视角），k 维度跨步为 32（因为半精度，2 字节我们可以激进一点 load 32 个 k 了），256 的 thread block size
- 同时做了一个 2d 的 2x4 warp tiling，划分为 64x128 上下两半的 c 分块，一个 warp 负责 64x32，还是对应这 m16n8k16，4x4 轮 (k 维度另算）。
- tiling size 选择的策略有许多考量因素（具体还是参考我第一篇 SGEMM 的文章，谢谢）

代码：

```cpp
// a block calculate c[128][128]
template <const int BM = 128, const int BN = 128, const int BK = 32, typename T>
__global__ void hgemm_naive_kernel(T *a, T *b, T *c, int m, int n, int k) {
    // grid swizzling
    int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8; // 将执行块设置为 8 的宽度

    int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    int by = (linear_id / SWIZZLE_W) % gridDim.y;

    int tid = threadIdx.x; // 0~255
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // 搬运映射
    int load_a_row = tid / 4;        // 0~63
    int load_a_col = (tid % 4) * 8;  // 0,8,16,24
    int load_b_row = tid / 16;       // 0~15 (K 维度）
    int load_b_col = (tid % 16) * 8; // 0,8,16 ... 120 (N 维度）

    // A/B 都行优先
    __shared__ T As[BM][BK];
    __shared__ T Bs[BK][BN];

    // warp tiling
    // 每个 warp 负责  64 x 32 的 C 矩阵块
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // 寄存器总量：M 维 4 块 * N 维 4 块 * 每块 4 个寄存器 = 64
    float sum[4][4][4] = {0.f};

    // 主循环
    for (int bk = 0; bk < k; bk += BK) {

        // 1. cp.async load A
        uint32_t smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row][load_a_col]));
        uint32_t smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row + 64][load_a_col]));

        T *global_a0 = &a[(by * BM + load_a_row) * k + bk + load_a_col];
        T *global_a1 = &a[(by * BM + load_a_row + 64) * k + bk + load_a_col];

        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);

        // 2. cp.async load B
        uint32_t smem_b0 = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[load_b_row][load_b_col]));
        uint32_t smem_b1 = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[load_b_row + 16][load_b_col]));

        T *global_b0 = &b[(bk + load_b_row) * n + bx * BN + load_b_col];
        T *global_b1 = &b[(bk + load_b_row + 16) * n + bx * BN + load_b_col];

        CP_ASYNC_CG(smem_b0, global_b0);
        CP_ASYNC_CG(smem_b1, global_b1);

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();

        // 3. Tensor Core 计算阶段 (K 分 2 步，一次 16 个 k)
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 16;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // 4 次 ldmatrix A (4 * 16 = 64 行）
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
                // ldmatrix x4 读 16x16
                int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
                int a_col = k_offset + (lane_id / 16) * 8;
                uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&As[a_row][a_col]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

            // 4 次 ldmatrix B (4 * 8 = 32 列）
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                // Lane 0~15 的线程恰好覆盖了 16 行 （两块 8x8 的首地址）
                int b_row = k_offset + (lane_id % 16);
                int b_col = warp_id_n * 32 + n_idx * 8;

                uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[b_row][b_col]));
                LDMATRIX_X2_TRANS(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
            }

            // MMA 核心运算：4x4 次 m16n8k16
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < 4; ++n_idx) {
                    if constexpr (std::is_same_v<T, __half>) {
                        M16N8K16_F16(sum[m_idx][n_idx][0],
                                     sum[m_idx][n_idx][1],
                                     sum[m_idx][n_idx][2],
                                     sum[m_idx][n_idx][3],
                                     reg_a[m_idx][0],
                                     reg_a[m_idx][1],
                                     reg_a[m_idx][2],
                                     reg_a[m_idx][3],
                                     reg_b[n_idx][0],
                                     reg_b[n_idx][1]);
                    } else {
                        M16N8K16_BF16(sum[m_idx][n_idx][0],
                                      sum[m_idx][n_idx][1],
                                      sum[m_idx][n_idx][2],
                                      sum[m_idx][n_idx][3],
                                      reg_a[m_idx][0],
                                      reg_a[m_idx][1],
                                      reg_a[m_idx][2],
                                      reg_a[m_idx][3],
                                      reg_b[n_idx][0],
                                      reg_b[n_idx][1]);
                    }
                }
            }
        }
        __syncthreads();
    }

    // ---------------- 写回 C 矩阵 ----------------
    int t_row = lane_id / 4;       // 0~7
    int t_col = (lane_id % 4) * 2; // 0, 2, 4, 6

#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int c_base_row = by * BM + warp_id_m * 64 + m_idx * 16;
            int c_base_col = bx * BN + warp_id_n * 32 + n_idx * 8;
            int idx_0 = (c_base_row + t_row) * n + c_base_col + t_col;
            int idx_2 = (c_base_row + t_row + 8) * n + c_base_col + t_col;
            if constexpr (std::is_same_v<T, __half>) {
                HALF2(c[idx_0]) = __float22half2_rn(FLOAT2(sum[m_idx][n_idx][0]));
                HALF2(c[idx_2]) = __float22half2_rn(FLOAT2(sum[m_idx][n_idx][2]));
            } else {
                BFLOAT2(c[idx_0]) = __float22bfloat162_rn(FLOAT2(sum[m_idx][n_idx][0]));
                BFLOAT2(c[idx_2]) = __float22bfloat162_rn(FLOAT2(sum[m_idx][n_idx][2]));
            }
        }
    }
}
```

## 2. hgemm_bcf

很显然 naive 版的代码，As/Bs 的读取没有经过 swizzle，肯定是有 bank 冲突的。现在我们来优化共享内存访问。

之前 tf32 的文章中，我们已经有了两个 swizzle 宏函数：

```c++
#define SWIZZLE_A(row, col) ((col) ^ (((row >> 1) & 0x3) << 2))

#define SWIZZLE_B(row, col) ((col) ^ (((row) & 0x7) << 3))
```

那么这俩 swizzle 函数能不能直接拿到半精度的 kernel 里用？显然不能，因为我们现在处理的是半精度数据，As 和 Bs 的 layout 是 128x32 和 32x128，那么 As 的 col 会出现 0~31，Bs 的 row 也是。

但把这作为起点，从 tf32 的 swizzle 延伸到半精度的推导却非常简单。因为仔细考虑一下，半精度占两个字节，而原来 tf32 是 4 个字节。那么就会导致原来 16 字节对齐的 col 下标，现在会翻倍。
也就是说，原来我们只需要保证 col 的低 bit0~1 不被扰动即可，现在需要保证 bit0~2 不被扰动，那么显然 SWIZZLE_B 本来就满足要求，所以可以直接使用，但是 SWIZZLE_A 就需要修改了，改动很简单乘个 2 即可。

由此，我们得到 As/Bs 读写无冲突的两个 swizzle 函数：

```cpp
#define SWIZZLE_A(row, col) ((col) ^ (((row >> 1) & 0x3) << 3))

#define SWIZZLE_B(row, col) ((col) ^ (((row) & 0x7) << 3))
```

代码核心修改：

```cpp
// cp.async load A
uint32_t smem_a0 =
    static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row][SWIZZLE_A(load_a_row, load_a_col)]));
uint32_t smem_a1 = static_cast<uint32_t>(
    __cvta_generic_to_shared(&As[load_a_row + 64][SWIZZLE_A(load_a_row + 64, load_a_col)]));

...

// cp.async load B
uint32_t smem_b0 =
    static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[load_b_row][SWIZZLE_B(load_b_row, load_b_col)]));
uint32_t smem_b1 = static_cast<uint32_t>(
    __cvta_generic_to_shared(&Bs[load_b_row + 16][SWIZZLE_B(load_b_row + 16, load_b_col)]));

// 读取
...
uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&As[a_row][SWIZZLE_A(a_row, a_col)]));

...
uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[b_row][SWIZZLE_B(b_row, b_col)]));
```

## 3. hgemm_bcf_dbf

上一版代码在保障 16 字节对齐的情况下解决了 bank conflict。那么直接加上双 buffer 流水线（具体流程参考我第一篇文章）
代码也不贴了，下一小节填坑重点说一下。

## 4. hgemm_bcf_dbf_rw

我们之前 tf32 的 kernel 中就提了，还有 c 矩阵写回 global memory 没有做到事务合并。但其实是可以做的，只是当时偷懒没写。现在正好前置步骤都已经理顺了，上面的过程也不费脑子，我们干脆把这一步也补上。

核心思想很简单：利用 As/Bs 的共享内存作为中转 Buffer，让所有线程把 fragment c 的结果先写入 smem 排整齐，最后再用 float4 向量化指令读取并一把梭写回 gmem。

这里有一个极其巧妙的巧合：As 和 Bs 占用的共享内存大小是 2 *(128*32 + 32*128)* 2B = 128*128*2），这正正好能放得下一个 Block 计算出的 128x128 的 C 矩阵块！如果是存不下，还要分批中转，那我可能就真不想写了（笑）。

在动手之前，我们要重点理解一下 fragment C 的寄存器状态。mma.sync.m16n8k16 指令计算后，输出的是一个 16x8 的小矩阵，我们需要搞清楚每个线程究竟 hold 住了这个矩阵里的哪些值。NV 官方文档有个图，mma m16n8k16 的计算 shape 下，fragment c 为：

![](static/fragment_c.png)

为了更直观，我也画了一张图：

![](static/my_fragment_c.svg)

总结下来规律很清晰：在一个 warp 内，每 4 个线程负责同一行的 8 个元素（fp16/bf16），跨过 8 行之后，这个排布再重复一次。
具体到线程级别，每个线程手里攥着 4 个值。比如：

- T0 拿着 c[0][0], c[0][1] 和跨 8 行的 c[8][0], c[8][1]
- T1 拿着 c[0][2], c[0][3], c[8][2], c[8][3]
- ... 以此类推

当然，别忘了我们外层还有一个 4(m)x4(n) 的循环，这意味着在 C 矩阵的全局坐标系下，每次迭代其实是在跨越 16 行或 8 列。

好，理理解了寄存器分布，现在的任务就是：把 T0~T31 手里的数据，同行同列地拼凑起来，写进我们那个 128x128 的 smem 里。

但是，前方 bank conflict 预警！

一个 warp 内，每 4 个线程写一行，总共覆盖 8 行。而在我们这个 128 宽度的 smem 里，显然，不同行的同一列，在物理地址上是绝对对齐的。如果直接写，这 8 行的线程会瞬间撞在同一个 Bank 上，引发极其惨烈的 8-way bank conflict！

不过，要错开 Bank 也很简单，我们直接把 B 矩阵的 Swizzle 宏拿过来“复用”即可。
为什么？以 warp 0，m/n offset 都等于 0 为例：

```yaml
row: 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7
col: 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6
```

![](static/swizzle_c.svg)

仔细观察二进制，row 是 0~7，即 00xxx，有效变量位是 bit0~2，然后我们还要保障 16 字节对齐，所以直接取了低 3bits，左移三位正好和 col 的低 3bit 错开，异或后即可遍历所有 32 个 bank。

>哎等等，有人说，这里左移三位那不是越到了 bit5，超过 32 了，感觉不对呀。

同志，请注意，由于我们用的是 fp16/bf16 元素的下标，半精度只占两字节，所以 bank id 的计算公式是 (row*128 + col)*2 / 4 % 32 = (col/2) % 32，这里本来就要看 6 个 bits 的，或者说 bit5 还是会被右移回 bit4，嘿嘿~）

其他 warp，offset 情况，同理即可。由此，我们就用极其优雅的位运算，把 c 矩阵零冲突地写进了共享内存里了。

核心修改如下：

```cpp
#define SWIZZLE_C(row, col) ((col) ^ (((row) & 0x7) << 3))

...
// A/B 都行优先，用 union 复用同一块内存，写法优雅
__shared__ __align__(128) union {
    // 前半段计算用的 A 和 B
    struct {
        T As[2][BM][BK];
        T Bs[2][BK][BN];
    };
    // 后半段写回用的 C
    T Cs[BM][BN];
} smem;

...

// 复用 As/Bs 中转
__syncthreads();

int t_row = lane_id / 4;       // 0~7
int t_col = (lane_id % 4) * 2; // 0, 2, 4, 6 每四个线程负责一行的 8 列 共 16 字节

// register to Cs smem
#pragma unroll
for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
    for (int n_idx = 0; n_idx < 4; ++n_idx) {
        int c_base_row = warp_id_m * 64 + m_idx * 16; // m 跨 16 行
        int c_base_col = warp_id_n * 32 + n_idx * 8;  // n 跨 8 列

        // 16 行我们分成两次 8 行写入
        int c_row_0 = c_base_row + t_row;
        int c_row_2 = c_base_row + t_row + 8;
        int c_col = c_base_col + t_col;

        if constexpr (std::is_same_v<T, __half>) {
            HALF2(smem.Cs[c_row_0][SWIZZLE_C(c_row_0, c_col)]) = __float22half2_rn(FLOAT2(sum[m_idx][n_idx][0]));
            HALF2(smem.Cs[c_row_2][SWIZZLE_C(c_row_2, c_col)]) = __float22half2_rn(FLOAT2(sum[m_idx][n_idx][2]));
        } else {
            BFLOAT2(smem.Cs[c_row_0][SWIZZLE_C(c_row_0, c_col)]) = __float22bfloat162_rn(FLOAT2(sum[m_idx][n_idx][0]));
            BFLOAT2(smem.Cs[c_row_2][SWIZZLE_C(c_row_2, c_col)]) = __float22bfloat162_rn(FLOAT2(sum[m_idx][n_idx][2]));
        }
    }
}

__syncthreads();

// smem to gmem
// 每个线程负责搬运 64 个元素 (fp16/bf16)，即 8 个 float4，256 个线程一次写 256*4*4 = 4096 字节
T *c_block = &c[by * BM * n + bx * BN];

#pragma unroll
for (int step = 0; step < 8; ++step) {
    // 保证同一个 warp 的 32 个线程，此时读取的 elem_idx 是绝对连续的
    int elem_idx = (step * 256 + tid) * 8;
    int row = elem_idx / 128;
    int col = elem_idx % 128;

    int s_col = SWIZZLE_C(row, col);

    FLOAT4(c_block[row * n + col]) = FLOAT4(smem.Cs[row][s_col]);
}
```

备注：这里之所以 FLOAT4 强转写回，得益于我们前面推导的 `SWIZZLE_C` 保留了 16 字节对齐特性。

## 5.benchmark、ncu report 和分析

话不多说，直接上 benchmark 结果和 ncu report

```yaml
n: 4096, m: 4096, k: 4096
torch                                    mean time: 4.097551 ms, 33.54 tflops
hgemm_cublas                             mean time: 4.210246 ms, speedup: 0.97, tflops: 32.64
hgemm_naive                              mean time: 5.191345 ms, speedup: 0.79, tflops: 26.47
hgemm_bcf                                mean time: 4.336920 ms, speedup: 0.94, tflops: 31.69
hgemm_bcf_dbf                            mean time: 4.096174 ms, speedup: 1.00, tflops: 33.55
hgemm_bcf_dbf_rw                         mean time: 4.075860 ms, speedup: 1.01, tflops: 33.72
```

![](static/hgemm_final.png)

### 一些讨论

- 老规矩，看一下 cuBLAS 的 kernel，`void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_nn_align8>(T1::Params)`
  - 256x64 的 tiling（m=256，n=64），32 的 k 维度切分，4 级流水线。
- 我的 ncu summary 还提示 `Uncoalesced Shared Accesses`, 具体原因是 cp.async + swizzle 之后，源/目标地址的连续性/单调性不满足其要求，导致 cp.async 需要 replay wavefronts 重复写入
  - 我最初其实是很想把它优化掉的，期望写一个完美的 kernel。刚开始不理解为什么会 Uncoalesced，于是我先尝试改造swizzle映射，无果；然后又尝试将 swizzle 改到读取 global memory，平铺写入 smem，再 swizzle 读出。可依然解决不了这个问题。
  - 经查阅资料和拷打 Gemini 得知：一个 warp 的 32 个线程执行 cp.async 搬运512B数据时，理想情况下，LSU 会将其打包为 4 次完美的 128B 内存事务（每次 8 个线程恰好填满一行 Smem）。但由于我做了 swizzle，原本这 8 个线程连续的写入地址被打散。这些散乱的地址既不连续也不单调递增，crossbar 无法在一个时钟周期内把数据路由到对应的 Bank 里，只能将事务拆解，从而触发了多次 wavefronts 写入。
    - 备注：这块底层的微架构行为我并不确定是否描述正确且严谨，欢迎懂的大佬在评论区指正补充，谢谢
- 为什么 cuBLAS 没有这个问题
  - 我不知道 cuBLAS 具体实现是什么样的，但它肯定有它的代价，点开它的 shared memory 统计表，可以发现高达 50 多万次 bank conflict！而我，0 冲突（实际约有 0.14%冲突，这是不同 warp 间冲突导致，可忽略）
  - 这其实是极致性能调优中的 Trade-off：我选择了绝对无冲突的读，牺牲了 cp.async 写的合并性；而 cuBLAS 保底了异步拷贝写入的极速，容忍了计算读取时产生的部分冲突。
  - cuBLAS 和我选择了不同的妥协方向，孰优孰劣，欢迎朋友们评价一下

## 6. 结束

本文应该是纯血 gemm 系列最后一篇了，从 fp32 到 tf32，再到半精度，从 naive 到超越 cuBLAS，整个过程相信大家对 GPU 架构、CUDA 编程、性能优化有了更深刻的理解。

如有错误，请大家指正。完整 kernel 和测试代码可以从 github 获取，欢迎大家关注我的手撕算子系列 vitamin-cuda 项目：

以上。
