# [CUDA 优化实战] sgemm tf32 - 超越 cuBLAS：Tensor-core、cp.async、ldmatrix、mma

## 0. 序 - 向量化计算的时代

（干货核能预警，大量配图，涉及硬核的 swizzle 推导过程，layout 地址坐标映射分析，指令说明，建议在 PC 端阅读以获得最佳体验）

（本文适用于有一定 CUDA 编程基础，熟悉 GEMM 优化，对进阶 tensor core / 嵌入 PTX 指令 性能调优感兴趣的读者阅读）

（所有 kernel 完整代码可以从 github 获取，欢迎大家关注我的手撕算子系列 vitmin-cuda 项目：）
>
> 怎么感觉要写出超越 cuBLAS 系列合集呢
>
> 在如今 Tensor Core 满天飞的时代，如果你还不知道怎么用 Tensor Core 进行 GEMM 计算，那你可能已经落后于时代了。

本文以 M=N=K=4096（MxKxN, cuBLAS 最擅长的中等规模）的 GEMM tf32 为例，在 RTX 5060 移动版显卡上，本人将使用 cp.async、ldmatrix、mma 等 PTX 指令，配合 Tensor Core TF32 加速计算，在同精度赛道上成功超越了 NVIDIA 原厂的 cuBLAS。本文将复盘这场与 cuBLAS 较量（是真较量，用 ncu 一步步 profile + 迭代优化出来的）过程。细节会具体到 cp.async 指令、ldmatrix/mma warp 级别 PTX 指令的运用，极其硬核的 swizzle 推导（虽然最后没全用上）, layout 分析等等

本文会给出 5 个 kernel 实现，从基础的 cp.async + 双 ldmatrix + mma，用 swizzle 解决 bank conflict，逆推 cuBLAS 策略更进一步优化 smem 访问，使用 grid swizzling 复用 L2，double buffer 隐藏时延 等手段完成最终对 cuBLAS 的超越。整个过程涉及对指令要求/用法的介绍和分析，精巧的 swizzle 设计，希望能帮助读者深入理解 Tensor Core 的使用和优化技巧。

同之前 SGEMM kernel 相比不太一样，之前本人是先有的优化路径然后写的几版代码（个人比较熟悉 gemm 优化手段），但由于本人对 tensor core 也并不熟悉，这一次是 ncu profile 驱动型演进。并且我先 profile 了 cuBLAS 的 kernel，从他的 kernel 名字和 shared memory table 结果逆推+参考其优化策略，最终实现了性能超越。（本人有合理理由怀疑 cuBLAS 应该是很久未更新，或者说未针对消费级显卡进行优化，因为本人在还有优化策略没全用上的时候就已经匹配上了 cuBLAS 的性能）

kernel 大纲如下（第一个是 cuBLAS kernel）

- sgemm_cublas tf32 版
- sgemm_tf32_bt（向量化读 A/B，B 转置写入 smem, 双 ldmatrix + mma）
- sgemm_tf32_bt_swizzle （向量化读 A/B，B 转置写入 smem, 双 ldmatrix + mma, As 0 冲突）
- sgemm_tf32_bt_swizzle_dbf （向量化读 A/B，B 转置写入 smem, 双 ldmatrix + mma, As 0 冲突，grid swizzling, 97~102% cuBLAS 性能）
- sgemm_tf32_swizzle_bcf (cp.async A/B，swizzle， As/Bs 无冲突，grid swizzling)
- sgemm_tf32_swizzle_bcf_dbf (cp.async A/B，swizzle， As/Bs 无冲突，grid swizzling，双 buffer，超越 cuBLAS)

## 1. PTX 指令 && ncu profile cuBLAS kernel

当我准备开始写 sgemm tf32 kernel 时，先去搜索了下 ldmatrix、mma 指令相关的博客文章想好好学一下。大概翻了下发现，大多讲的是半精度 GEMM（倒也合理，毕竟是 Tensor Core 最初就是为半精度设计的），但是难道大家都是从半精度学起的，就不管 tf32 了嘛（我丢）。无奈，只好自己开始翻 nvidia 的 PTX 文档，以及 profile 一下 cuBLAS kernel 作为参考。

### PTX 指令

PTX（parallel-thread-execution）官方文档：<https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>，是 nv 提供的虚拟汇编指令语法集。在本文中主要用到的指令有：cp.async、ldmatrix、mma。cp.async 是异步拷贝指令，ldmatrix/mma 是 warp 级别协同搬运/计算指令。

具体讲解指令之前，先说明一下在 C++ (CUDA) 代码中嵌入 PTX 汇编指令的语法结构。它使用的是 GCC 扩展内联汇编（Extended Asm）语法，一般为：

```c++
asm volatile(
    "汇编指令模板；"
    : 逗号分隔的输出操作数   /* 可选 */
    : 逗号分隔的输入操作数   /* 可选 */
    : 破坏描述符 (Clobbers) /* 可选 */
);
```

比如我们即将用的到 cp.async:

```cpp
asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(dst_smem_32b), "l"(src_global_ptr))
```

核心关键字解析：

- asm：内联汇编的关键字，告诉编译器这里开始是一段汇编代码。
- volatile：告诉编译器不要对这段代码进行优化（比如不要随意改变执行顺序，或者认为没有使用其输出就将其删除）。

标点与占位符解析：

- %0, %1：这些是汇编字符串里的占位符。编译器会按照下面输出和输入操作数列表的顺序，从 0 开始依次将 C++ 变量映射到这些占位符上。
- 换行符 \n：这纯粹是为了排版。当编译器把 C++ 编译成汇编文件时，加了 \n 能保证生成的汇编代码优雅地换行。
- 冒号 : 与 :: 用法：冒号用于分隔汇编代码、输出、输入等部分。如果一条指令只有输入，没有输出，为了让编译器知道后面的参数是输入，就必须用两个冒号 :: 跳过输出部分。
- 破坏描述符是开发者主动告诉编译器，该汇编指令会影响特定的物理寄存器、内存或状态标志位中的值，以此强制编译器放弃相应的 cache，而重新读取，避免后续复用时出现严重的逻辑错误。

约束字符 (Constraints)，在绑定 C++ 变量和汇编操作数时，我们需要用字符串告诉编译器该如何分配寄存器：

- = （修饰符）：表示“只写”，通常用于输出操作数。如果不带 =，则默认是“只读”，用于输入操作数。
- r (Register)：表示将变量放入一个 32 位的通用整数寄存器中。
- l (Long)：表示 64 位寄存器（在 CUDA 中通常用来存储 64 位全局内存指针，比如 src_global_ptr）。
- f (Float)：表示 32 位浮点寄存器。
- "=r"：将结果输出到一个 32 位通用寄存器。
- "r"：从一个 32 位通用寄存器中读取输入。
- "=f","f": 同理

好了，下面介绍下我们即将用到的指令，只说我们用到指令的具体用法，输入输出列表也省了，免得太冗长枯燥，各指令的其他用法还是请参考官方文档

```cpp
cp.async.cg.shared.global.L2::128B [%0], [%1], 16; // cg： (Cache at Global level)：bypass L1，拷贝 16 字节
cp.async.commit_group;  // 提交异步拷贝任务
cp.async.wait_group 0;  // 表示允许当前线程后台异步的 group 数，0 表示不允许后台，要等待到全部完成
```

- `cp.async`：异步拷贝指令，支持 bypass L1/register，从 gmem 到 L2 直达 smem，节省寄存器资源，并且异步性非常适合与计算重叠作为流水线实现的基础

```cpp
mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32        d, a, b, c;
```

- `mma`指令：介绍 ldmatrix 之前要先说 mma，因为 ldmatrix 就是为 mma 服务的
  - mma 输入的 AB 子矩阵，A 为行优先矩阵，B 为列优先矩阵。
  - 根据官方文档，我们这个后来才加入 tf32 精度计算，只支持 shape m16n8k8
  - shape 定了后，输入的 fragment A,B,C（后面会详细说）的 shape 和 下标值也确定了
总之，mma 是有一点死板（个人感觉）

```cpp
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4]; //协同加载 4 个 8x8 矩阵，每个线程最终 hold 4 个值
ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2]; //协同加载 2 个 8x8 矩阵，每个线程最终 hold 2 个值
```

- ldmatrix：一个 warp 从 smem 协同加载一个小矩阵分块到所有线程，所有线程一起 hold 着的寄存器结果叫做一个 fragment （死板+1）
  - 注意：x2，x4，表示读取线程数为 16，32。ldmatrix 读取数据，提供地址的每个线程永远是读 16 字节！读取完后会分发到各个线程，组成一个 fragment。理解这一点，才能理解如何解决 bank conflict
  - 由于 tf32 mma 只支持 m16n8k8，为了加载 16x8 的 A 和 8x8 的 B 矩阵，所以我们只能用 m8n8 shape + .b16 类型去搬运 32bit 的数据
  - 同样由于 tf32，我们也无法使用 .trans 转置（这也是个坑，手动转置和尝试寄存器 shuffle 转置折腾了我很久）

### cuBLAS kernel ncu report

![](static/cublas_tf32_ncu_0.png)

![](static/cublas_tf32_ncu_1.png)

说实话，第一眼看到 cuBLAS 的 ncu report 时，我是有点发虚的。compute/memory 吞吐不低，完美的 shared memory 统计表，使用了 cp.async，ldmatrix，shared load，还做到了 0 个 bank conflict！再加上全合并的 global memory 访问，四级流水线，单线程寄存器更是被狠命压榨到了 228 个。看起来似乎没有优化空间了，这如何赶上它的性能啊。所以最初，心想着能达到 95% 性能就差不多了。

再看一眼 kernel 名字，调用的 cutlass kernel `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_64x256_16x4_nn_align4>(T1::Params)`, 嗯，mma m16n8k8，64x256 的 tiling（m=64，n=256），16 的 k 维度切块，4 级流水线。

ok, 初步摸清对手底细了，开始手搓我们自己的 kernel

## 2. sgemm_tf32_bt

这个初版 kernel，我的想法很简单，就是把 cp.async 用上，ldmatrix + mma 启动起来，就算完事。其他 bank conflict，合并访存什么的就先不管了。虽然大概猜出 cuBLAS 的 cutlass 的实现，但是我并不想照抄他的。抄，抄还能超得过师傅吗，何况逆推的策略也不一定准确完整。4 级流水线我也不想写（要改 smem 大小），然后进行 4 个 stage 的调度，我表示放弃。

![](static/tiling.svg)

当然，也没有那么粗糙。我还是按着原来 SGEMM kernel 的思路

- 128x128 的 tiling（c 矩阵视角），k 维度跨步为 16，256 的 thread block size（tiling size 选择的策略有许多考量因素，详细情况见我上一篇文章）
- 同时做了一个 2d 的 2x4 warp tiling，划分为 64x128 上下两半的 c 分块，一个 warp 负责 64x32，对应 mma 的 16x8x8，正好 64/16=4，32/8=4，共 4x4 轮（k 维度累加另算），完美（我就是喜欢正方形，对称就是好）。

然后整个流程也没有太多要点：

- As 矩阵用 cp.async 从 global memory bypass L1 直达 smem，Bs 矩阵用 LDG + 手动转置写入 smem
- 然后用 ldmatrix 加载 As，Bs 到寄存器， mma 计算
- 得到 c fragment，写回 c

唯一要先提一下的是，mma 的结果 c fragment 的布局（就是一个 warp 内每个线程 hold 哪些值），不然你都不知道怎么映射回全局坐标，这个具体见：<https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688>

代码：

```cpp
// a block calculate c[128][128]
template <const int BM = 128, const int BN = 128, const int BK = 16>
__global__ __launch_bounds__(256, 2) void sgemm_tf32_bt_kernel(float *a, float *b, float *c, int m, int n, int k) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x; // 0~255
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // 搬运映射
    int load_a_row = tid / 4;               // 0~63
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12
    int load_b_row = tid / WARP_SIZE;       // 0~7  (K 维度）
    int load_b_col = (tid % WARP_SIZE) * 4; // 0~124 (N 维度）

    // A 保持 行优先，B 转置为 列优先
    __shared__ float As[BM][BK];
    __shared__ float Bs[BN][BK];

    // 2x4 warp tiling
    // 一行 warp 负责上下 64x128， 每个 warp 负责  64 x 32 的 C 矩阵块
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // 寄存器总量：M 维 4 块 * N 维 4 块 * 每块 4 个寄存器 = 64
    float sum[4][4][4] = {0.f};

    // 主循环
    for (int bk = 0; bk < k; bk += BK) {

        // 1. 使用 cp.async 加载 A 矩阵 (16 bytes 对齐）
        uint32_t smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row][load_a_col]));
        uint32_t smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row + 64][load_a_col]));

        float *global_a0 = &a[(by * BM + load_a_row) * k + bk + load_a_col];
        float *global_a1 = &a[(by * BM + load_a_row + 64) * k + bk + load_a_col];

        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);
        // 提交所有的异步拷贝任务
        CP_ASYNC_COMMIT_GROUP();

        // 2. 加载 B 矩阵并手动转置写入 smem
        float4 tmp_b0 = FLOAT4(b[(bk + load_b_row) * n + bx * BN + load_b_col]);
        float4 tmp_b1 = FLOAT4(b[(bk + load_b_row + 8) * n + bx * BN + load_b_col]);

        // 将读取的 B 手动转置写入 smem
        Bs[load_b_col + 0][load_b_row] = tmp_b0.x;
        Bs[load_b_col + 1][load_b_row] = tmp_b0.y;
        Bs[load_b_col + 2][load_b_row] = tmp_b0.z;
        Bs[load_b_col + 3][load_b_row] = tmp_b0.w;
        Bs[load_b_col + 0][load_b_row + 8] = tmp_b1.x;
        Bs[load_b_col + 1][load_b_row + 8] = tmp_b1.y;
        Bs[load_b_col + 2][load_b_row + 8] = tmp_b1.z;
        Bs[load_b_col + 3][load_b_row + 8] = tmp_b1.w;

        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();

        // 3. Tensor Core 计算阶段 (K 维度走 2 步，每次消耗 8 个 K)
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 8;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // 发射 4 次 ldmatrix 获取 A 矩阵块 (4 * 16 = 64 行）
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
                // warp_id_m 跨度是 64
                int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
                int a_col = k_offset + (lane_id / 16) * 4;
                uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&As[a_row][a_col]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

            // 发射 4 次 ldmatrix 获取 B 矩阵块 (4 * 8 = 32 列）
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                // warp_id_n 跨度是 32
                int b_row = warp_id_n * 32 + n_idx * 8 + (lane_id % 8);
                int b_col = k_offset + ((lane_id / 8) % 2) * 4;
                uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[b_row][b_col]));
                LDMATRIX_X2(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
            }

            // MMA 核心运算：4x4 的 1688
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < 4; ++n_idx) {
                    M16N8K8(sum[m_idx][n_idx][0],
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
        __syncthreads();
    }

    // Tensor Core m16n8k8 C 寄存器碎片的标准排布映射法则
    // c fragments layout:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688 1688.tf32
    int t_row = lane_id / 4;       // 0~7
    int t_col = (lane_id % 4) * 2; // 0, 2, 4, 6

#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            // 根据新的 Warp 跨度重新计算 Global C 的基址
            int c_base_row = by * BM + warp_id_m * 64 + m_idx * 16;
            int c_base_col = bx * BN + warp_id_n * 32 + n_idx * 8;

            c[(c_base_row + t_row) * n + c_base_col + t_col] = sum[m_idx][n_idx][0];
            c[(c_base_row + t_row) * n + c_base_col + t_col + 1] = sum[m_idx][n_idx][1];
            c[(c_base_row + t_row + 8) * n + c_base_col + t_col] = sum[m_idx][n_idx][2];
            c[(c_base_row + t_row + 8) * n + c_base_col + t_col + 1] = sum[m_idx][n_idx][3];
        }
    }
}
```

跑个 benchmark 看一下：

```yaml
n: 4096, m: 4096, k: 4096
torch                                    mean time: 15.146454 ms, 9.07 tflops
sgemm_cublas_tf32                        mean time: 8.535476 ms, speedup: 1.77, tflops: 16.10
sgemm_tf32_bt                            mean time: 15.925327 ms, speedup: 0.95, tflops: 8.63
```

我去，惨不忍睹啊，比 pytorch 的 fp32 矩阵乘法还要慢。ncu profile 一下

![](static/bt.png)

醒目的 bank conflicts 比例（avg 32.6-way 冲突，都不知道算怎么出来的），不过也是意料之中了。接下来优化 smem 访问

## 3. sgemm_tf32_bt_swizzle

### As 优化

先说一下，数据是如何搬运到 As，以及 ldmatrix 是如何访问 As 的。128x16 行数据，cp.async 部分可以先忽略，因为类似于我们之前平铺线程 float4 向量化访问（区别在于这里我们 bypass 了 L1 和寄存器）
重点下需要理解一下 ldmatrix load As 的过程。通过阅读官方文档，我简单总结一下。

一个 warp 执行 ldmatrix 有两个阶段：

- 指定的读取线程从对应的地址读取 16 字节数据（x1，x2，x4，指定的线程数分别为前 8，16，32 个线程）
- 读取完的数据划分给 32 个线程

这里我们 warp tiling 划分得比较好，正好可以用 ldmatrix x4，即 32 个线程读取 16x8 个数据。

```cpp
int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
int a_col = k_offset + (lane_id / 16) * 4;
```

As 是 128x16 float 数组。从 As 读取 16x8 的矩阵，我们给了 16 行 2 列共 32 个地址（每个地址是连续 16 字节数据的首地址，两列之间相距 4 列，其实就是 16 行连续的 8 个 float），计算一下每个线程读取的地址：（以 warp 0, m_idx=0，k_offset=0 为例）

```yaml
row:0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
col:0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4
```

首先 32 线程访问 512bytes 是宽内存事务，gpu 会拆分为 4 个 phases，每 8 个线程一组对应一个内存事务周期，可以只看前 8 个线程，情况是：

```yaml
row:0, 1, 2, 3, 4, 5, 6, 7
col:0, 0, 0, 0, 0, 0, 0, 0
```

- 显然，隔行同列就会冲突，所以是 4-way bank conflict.

好，根据我们的习惯，开始推导 xor swizzle 公式，我们需要找到一个映射 f(row, col) --> (row, new_col)，避开这四个冲突，仔细观察：

- row 的二进制为 00000~01111(0xxxx)，前 8 个线程为 00000~00111
- col 完全相同，00000（只看前 8 线程）

![](static/swizzle_a.svg)

冲突是因为相同 bank id 的不同地址导致的，更进一步，我们这里是由于隔行的 col 值相同导致的（因为宽度是 16 个 bank，所以理解起来费劲一点儿），那 row 不同，我们就想办法用 row 的 bit 位去扰动 col。

首先，我们已经知道每两行是 32 个 bank，每两行一组也就是说 row 的 bit0 不起作用，把 row 的 bit0 抹为 0，把每两行绑定（绑定的意思是这两行最终的列偏移一致，这样能留出空间与其他行错开），所以有 `(row>>1)<<1`；

其次， cp.async 是 16 字节（即 float4）对齐，即我们不能扰动 col 的低 2bit。那我们能直接左移 1 位 用 `col^((row>>1) <<2)` 吗？不行，为什么？因为我们数组大小是 128x16，一行里 float4 对齐的地址只能是 0，4, 8, 12 四个值，而且 col 最大不能超过 15（即二进制里 1 不能左移到 bit5 及以上的位置，不然越界了），因此我们只能保留 row>>1 的两个有效位 bit0~1, 即 `((row >> 1) & 0x3)<<2`；

这样，row 的 bit1~2 被推到了 bit2~3，即 00xx0, 正好是 0，4，8，12 四个值。而 col 全相同，因此异或后，依然遍历了 bit2~3 的四种排列；
由此，我们得到了一个 float4 对齐的 As 读写无冲突版的 xor swizzle 公式：`new_col = col ^ (((row >> 1) & 0x3) << 2)`

m_idx、k_offset 的不同情况，以及三组 8 线程和其他 warp，同样适用。因为原理是一致的，只是个别 bit 位同步反转了，这里就不重复推导过程了。

有人会说，你这怎么感觉是在硬凑呢，col xor 上这堆奇奇怪怪的东西是什么含义。哎，你说对，就是硬凑，我根据 xor 的性质凑出来这个 xor swizzle，花了好久呢。要说具体含义，就是在满足硬件条件的前提下扰动 col 的 bit 位，使得其分布在 0，4，8，12 位置上，错开 bank。做 xor swizzle 就是这么个过程。

核心修改，如下：

```cpp
#define SWIZZLE_A(row, col) ((col) ^ (((row >> 1) & 0x3) << 2))

//cp.async
uint32_t smem_a0 =
            static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row][SWIZZLE_A(load_a_row, load_a_col)]));
uint32_t smem_a1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&As[load_a_row + 64][SWIZZLE_A(load_a_row + 64, load_a_col)]));

// ldmatrix x4
uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&As[a_row][SWIZZLE_A(a_row, a_col)]));
LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
```

### Bs 优化

看一下从 global memory 写入到 Bs。由于我们的转置操作，数组也是 128x16 布局（现在看到这个 layout 就 PTSD），并且还是标量写入

```cpp
int load_b_row = tid / WARP_SIZE;       // 0~7  (K 维度）
int load_b_col = (tid % WARP_SIZE) * 4; // 0~124 (N 维度）
```

根据代码，实际每个线程写入地址为：（以 warp 0 为例）

```yaml
row:0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124
col:0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
```

一看，就知道是爆炸的 32-way 冲突...

那读取 Bs smem 呢，从 Bs 用 ldmatrix x2 读取 8x8 的矩阵，我们给了 8 行 2 列共 16 个地址（类似 As，8 行连续的 8 个 float）

```cpp
int b_row = warp_id_n * 32 + n_idx * 8 + (lane_id % 8);
int b_col = k_offset + ((lane_id / 8) % 2) * 4;
```

计算一下每个线程读取的地址，以 warp0 ，n_idx 为 0，k_offset 为 0 为例：（ 只看前 16 线程，后半 16 线程不直接读取 smem）

```yaml
row:0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7
col:0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4
```

依然是宽内存事务请求，会展开 2 个 phases，8 线程一组，可以只看前 8 个线程，同理 4-way bank conflict.

看到这里我其实有点弃疗了，一门心思两头堵，感觉根本不可能同时解决 ldg 和 ldmatrix 的冲突。但是无奈，只能硬着头皮，先从简单的看起。

先看 ldmatrix 读取，这个比较好解决，ldmatrix x2 相当于 之前 ldmatrix x4 As 的前 16 线程，所以 Bs 的读取模式相当于 As 读取模式的子集。那么 As 的 swizzle 可以直接拿过来用， Bs 的读取冲突就被优化掉了。

好，我们现在有 `new_col = col ^ (((row >> 1) & 0x3) << 2)`，还算不错，一步就把 Bs 读取干掉了

但是，这个 swizzle 对前面 ldg 写入管用吗？仔细观察，ldg 写入时 row 的二进制为：

```yaml
00000，00100，01000，01100，10000，10100，11000，11100...
```

可以发现，实际上我们是取了 row 的 bit1~2 左移到 bit2~3 的位置，对于 0, 4, 8, 12 这些行号，异或的偏移量结果只有 0000 和 1000 两种状态。意味着32线程写入都撞在两个 bank 上，惨烈的16-way 冲突。

![](static/swizzle_b.svg)

那怎么办呢，再仔细观察，注意到 row 是 4 的整数倍，它的有效变量位其实在 bit2~4。如果考虑直接取 bit2~4 的话，那么有 `((row>>2)&0x7) <<2`。发现依然不行，因为首先数组直接越界了（推导 As 中也说过了），我们必须让扰动范围保持在 bit2~3，那么再限制一下就得到 `((row>>2)&0x3) <<2` 。

看起来不错，但这样有一个问题，`row>>2` 直接把四行打包了（四行的列偏移相同），这会使得 ldmatrix x2 产生 2-way 冲突。

怎么办，为了能够充分利用 row 的有效位 bit2~4，同时不影响 ldmatrix 的两行打包。我们必须想出一个方式，注意，接下来是黑魔法时刻：

- 取出 row 的 bit1~2 和 bit3~4 把他们异或，即 `((row >> 1) & 0x3) ^ ((row >> 3) & 0x3)` = `((row >> 1)  ^ (row >> 3)) & 0x3`,
- 然后再推到 bit2~3 的位置，`(((row >> 1) ^ (row >> 3)) & 0x3)<<2`

这样 ldg 写入冲突能降低到 8-way，因为已经完全遍历了 bit2~3 的四种排列；

ldmatrix x2 依然 0 冲突，`row>>1` 保障了两行打包，而 row 的高位 bit3~4 在 8 线程内值是恒定的，所以异或后即使 bit 反转（反转对 8 个线程来说是同时进行的）也不影响两行打包。

其他 n_idx，k_offset，warp 一样适用这个 swizzle，理由同上。到这，其实推导已经结束，没法更进一步。按我的理解，ldg 优化极限就是 8-way 冲突，因为数组是 16 个 float 的宽度，又有 float4 的对齐要求，那么只有 4 个偏移地址可用，优化极限就是 8-way 冲突。（当然我的理解可能有误，请大家指正。本人能力有限，只能推导到这里了）

也许还是有人会问，这一坨位运算异或的数学/物理含义到底是啥？怎么说呢，个人理解，我其实只是在利用 xor 的数学性质，实质上是提取 row 的有效变量位，扰动 col 的 bit，从而错开 bank。就结果而言，xor 的双射性保证了写入不会互相覆盖，其自反性 (row^col^col = row) 保证了读写同构，使用起来较为方便。我还是那个理解，希望大家对 xor swizzle 机制祛魅，理解 xor 性质，自己一样可以推导，重要的是理解用 row 的有效变量位去扰动 col 的 bit 这一基本原理。即使 cutlass 抽象出来所谓的 swizzle 模板，也只能用于特定数据类型的特定 layout。

现在，我们得到 Bs 的 swizzle 定义：`new_col = col ^ ((((row >> 1) ^ (row >> 3)) & 0x3)<<2)`

核心修改如下：

```cpp
#define SWIZZLE_B(row, col) ((col) ^ (((row >> 2) & 0x3) << 2))

...
Bs[load_b_col + 0][SWIZZLE_B(load_b_col + 0, load_b_row)] = tmp_b0.x;
Bs[load_b_col + 1][SWIZZLE_B(load_b_col + 1, load_b_row)] = tmp_b0.y;
Bs[load_b_col + 2][SWIZZLE_B(load_b_col + 2, load_b_row)] = tmp_b0.z;
Bs[load_b_col + 3][SWIZZLE_B(load_b_col + 3, load_b_row)] = tmp_b0.w;

Bs[load_b_col + 0][SWIZZLE_B(load_b_col + 0, load_b_row + 8)] = tmp_b1.x;
Bs[load_b_col + 1][SWIZZLE_B(load_b_col + 1, load_b_row + 8)] = tmp_b1.y;
Bs[load_b_col + 2][SWIZZLE_B(load_b_col + 2, load_b_row + 8)] = tmp_b1.z;
Bs[load_b_col + 3][SWIZZLE_B(load_b_col + 3, load_b_row + 8)] = tmp_b1.w;

...
uint32_t smem_addr =static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[read_idx][b_row][SWIZZLE_B(b_row, b_col)]));
LDMATRIX_X2(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
```

跑一下 benchmark：

```yaml
n: 4096, m: 4096, k: 4096
torch                                    mean time: 15.146454 ms, 9.07 tflops
sgemm_cublas_tf32                        mean time: 8.535476 ms, speedup: 1.77, tflops: 16.10
sgemm_tf32_bt                            mean time: 15.925327 ms, speedup: 0.95, tflops: 8.63
sgemm_tf32_bt_swizzle                    mean time: 9.786451 ms, speedup: 1.55, tflops: 14.04
```

嗯，效果还是很明显，还差 1~2ms 就接近 cuBLAS 了。

## 4. sgemm_tf32_bt_swizzle_dbf

通过之前的修改，我们做到了无 bank conflict 的 As 读写，Bs 读写冲突下降了许多。一时想不到办法解决 Bs 的冲突，而且我听传闻说 ldmatrix 走的是不同的硬件电路，和 ldg 不同。内心抱着意思侥幸心理，双 buffer 也许能掩盖这些冲突时延开销（毕竟之前 cuBLAS SIMT 也没解决冲突啊），所以我死马当作活马医，先上双 buffer 流水线看一下。流水线流程参考我的上一篇文章，详细代码也不贴了。直接跑一下 benchmark：

```yaml
n: 4096, m: 4096, k: 4096
torch                                    mean time: 15.146454 ms, 9.07 tflops
sgemm_cublas_tf32                        mean time: 8.535476 ms, speedup: 1.77, tflops: 16.10
sgemm_tf32_bt_swizzle_dbf                mean time: 9.025055 ms, speedup: 1.68, tflops: 15.23
```

![](static/cublas_l2.png)

![](static/L2.png)

更接近了，可惜还差 0.xms。继续对比 ncu 报告，我发现 cuBLAS 的 L2 cache 命中极高（85%+），我只有（30%+）。这个是不是有很大影响呢？

### Grid Swizzling

是不是 L2 cache 起的作用，试试就知道，上技巧 grid swizzle。grid swizzle 的意思就是把 block 访问 global memory tile 的顺序重新排列一下。默认情况下 cuda 是按照 grid 的  x （N）维度，再 y（M）维度顺序发射的。

![](static/grid_swizzle.svg)

我们手动调整一下 block 访问的实际分块，强行将遍历轨迹折叠成一个宽度为 8 的垂直长条。使得其在 x 维度上遍历 8 块后，就立刻向下进入下一行。完整一行 A/一列 B 矩阵 tile 大小为 128*4096*4 bytes = 2MB。如果不限制宽度，GPU 会先执行第 0 行的 32 个 Block。这 32 个 Block 共享同 1 个 A tile（2 MB），但它们分别需要读取 32 个不同的 B tile

也就是说，为了算完第 0 行的 Block，GPU 总共读取了 66 MB 的数据，这超过了我的 L2 cache 大小（32MB）！（说明在15个 block 后，L2 已经被 B tile 冲刷掉了）

限制 x 维度执行 8 块后，GPU 就会先执行完第 0 行的前 8 个 block，再执行第 1 行的前 8 个 Block，以此类推。

这样，每 8 个 Block 会读取 1 个 A tile（2 MB）和 8 个 B tile（16 MB），总大小为 18MB，小于 L2 cache 大小，可以完整驻留。在极短时间窗口内，发射的下一行 8 个 block 会读取新的 A tile，但是可以完全复用这 8 个 B tile，避免了重复读取 global memory 开销，吃满 L2 红利。

具体实现如下：

```cpp
    // grid swizzling
    int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8; // 将执行块设置为 8 的宽度

    int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    int by = (linear_id / SWIZZLE_W) % gridDim.y;
```

跑一下 benchmark 和 ncu：

```yaml
n: 4096, m: 4096, k: 4096
torch                                    mean time: 15.146454 ms, 9.07 tflops
sgemm_cublas_tf32                        mean time: 8.535476 ms, speedup: 1.77, tflops: 16.10
sgemm_tf32_bt_swizzle_dbf                mean time: 8.723189 ms, speedup: 1.83, tflops: 15.76
```

![](static/l2_opt.png)

果然有作用，时延又进步了些，ncu 显示我们 L2 cache 命中率提升到了 90%! 可惜，性能和 cuBLAS 比还差一点点。

## 5. sgemm_tf32_swizzle_bcf

就还差一点点了，shared memory 读写 swizzle，double buffer，grid swizzle 技巧都用上了，还怎么办。

我仔细观察，苦思冥想。重新翻开 cuBLAS 的 ncu profile 进行对比。寻找最大区别在哪。

![](static/cublas_sh.png)
![](static/sh.png)

cuBLAS 0 bank conflict，我 1.5+ 亿次冲突！这就是最大的区别，没理由 cuBLAS 能做到 0 冲突，我们做不到呀。cuBLAS 怎么做到的呢？通过分析对比 shared memory 统计表，我发现：

- cuBLAS 的 ldmatrix 指令数只有我 1/4，多了非常多的 shared load 指令；
- cp.async 指令数是我的两倍，ldg 指令为 0。

说明了什么，说明 cuBLAS 读取 global memory 全用的 cp.async，而有部分 smem 读取没有用 ldmatrix 指令，显然，就是 Bs 矩阵的读取使用了常规的 shared load 方法（为什么 ldmatrix 少了 3/4，这个可能和 cuBLAS 的 tiling size(64x256) 有关，他复用了更多次的 smem 读取和寄存器）。

既然逆推出了 cuBLAS 的 Bs 读取策略，那我们也用上吧，0 冲突的诱惑难以抗拒。我们把 Bs ldg 改成 cp.async，同时尝试把 ldmatrix 换成常规的 shared load。

这一步没有那么轻松，我们要理解 ldmatrix 后 b fragment 的详细状态，才能用直接读取 smem 的方式替换 ldmatrix，否则无法使用 mma 指令。通过查看官方文档

![](static/b_fragment.png)

仔细理解一下 m16n8k8 tf32 下 b fragment 的寄存器排布。如图，其中 b0，b1 表示每个线程的两个 32 位寄存器。load 一个 8x8 的矩阵后，其实是每四个线程拿着一列数据，T0 拿着的是 b[0][0],b[4][0]， T1 拿着的是 b[1][0],b[5][0]... T4 拿着的是 [0][1],[4][1]...

我也画个图，方便读者理解

![](static/thread_b_reg.svg)


这么一看，也不是很复杂（这里重点要理解，row/col是b 子矩阵真实的逻辑行列号，而与行优先/列优先的内存行列无关），完全可以从 global memory 读取数据原原本本放进 Bs，然后手动进行坐标映射去读取 Bs。
我们先把单 buffer 的 kernel 改掉。核心修改，去掉 Bs 转置变为 16x128，这个 layout 舒服多了，使用 cp.async 从 global memory 读取数据写入 Bs，然后直接读取 smem 进行坐标映射。

```cpp
#define SWIZZLE_B_F2(row, col) ((col) ^ (((row) & 0x7) << 3))

...

uint32_t smem_b0 =
            static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[load_b_row][SWIZZLE_B_F2(load_b_row, load_b_col)]));
uint32_t smem_b1 = static_cast<uint32_t>(
    __cvta_generic_to_shared(&Bs[load_b_row + 8][SWIZZLE_B_F2(load_b_row + 8, load_b_col)]));

float *global_b0 = &b[(bk + load_b_row) * n + bx * BN + load_b_col];
float *global_b1 = &b[(bk + load_b_row + 8) * n + bx * BN + load_b_col];

CP_ASYNC_CG(smem_b0, global_b0);
CP_ASYNC_CG(smem_b1, global_b1);

...
//内层循环中读取 Bs
#pragma unroll
for (int n_idx = 0; n_idx < 4; ++n_idx) {
    // 当前处理的 N 维度的基础列号
    int n_base = warp_id_n * 32 + n_idx * 8;

    // 每四个线程一列
    int b_col = n_base + (lane_id / 4);

    // k 维度的 0~3 行 和 4~7 行
    int b_row_0 = k_offset + (lane_id % 4);
    int b_row_1 = k_offset + (lane_id % 4) + 4;

    // swizzling 读取
    reg_b[n_idx][0] = __float_as_uint(Bs[b_row_0][SWIZZLE_B_F2(b_row_0, b_col)]);
    reg_b[n_idx][1] = __float_as_uint(Bs[b_row_1][SWIZZLE_B_F2(b_row_1, b_col)]);
}
```

注意，这里我们又引入了一个全新的 swizzle 宏 SWIZZLE_B_F2。具体枯燥的位运算推导过程就不在这里展开了（原理和前面类似，大家可以自己推一下试试）。核心设计目的其实就是满足两个条件：

- 满足 cp.async 的 16 字节对齐要求
- 四线程同列读取错开8个bank

benchmark 结果：

```yaml
n: 4096, m: 4096, k: 4096
torch                                    mean time: 15.146454 ms, 9.07 tflops
sgemm_cublas_tf32                        mean time: 8.535476 ms, speedup: 1.77, tflops: 16.10
sgemm_tf32_bt_swizzle_dbf                mean time: 8.723189 ms, speedup: 1.83, tflops: 15.76
sgemm_tf32_swizzle_bcf                   mean time: 8.650843 ms, speedup: 1.83, tflops: 15.89
```

起飞，单 buffer 实现甚至比双 buffer 还快了

![](static/sh_opt.png)

bank conflict 也降低为 0（实际还显示冲突 48618/100715659 ~=0.4%, 具体原因是不同 warp 间的冲突，这个在我上篇文章说过，可以忽略）

## 6. sgemm_tf32_swizzle_bcf_dbf

好，再给我们的无冲突版加上 double buffer 流水线优化，就是目前最终版的完全体 kernel 。最后贴一下完整 benchmark 和 ncu summary

```bash
n: 4096, m: 4096, k: 4096
torch                                    mean time: 15.146454 ms, 9.07 tflops
sgemm_cublas_tf32                        mean time: 8.535476 ms, speedup: 1.77, tflops: 16.10
sgemm_tf32_bt                            mean time: 15.925327 ms, speedup: 0.95, tflops: 8.63
sgemm_tf32_bt_swizzle                    mean time: 9.786451 ms, speedup: 1.55, tflops: 14.04
sgemm_tf32_bt_swizzle_dbf                mean time: 8.723189 ms, speedup: 1.83, tflops: 15.76
sgemm_tf32_swizzle_bcf                   mean time: 8.650843 ms, speedup: 1.83, tflops: 15.89
sgemm_tf32_swizzle_bcf_dbf               mean time: 8.275736 ms, speedup: 1.92, tflops: 16.61
```

![](static/final.png)

从 15.925327 到 8.275736，又是一次纯手搓的性能优化，正面硬刚并超越 cuBLAS ！这个过程中，我们通过精确的资源分配，结合 ldmatrix/mma 而设计的复杂精巧 (a,b,c 各不相同）的搬运/计算坐标映射，grid swizzle 拉满 L2 利用率，加上双 buffer 流水线技术，还逆推了 cuBLAS 的策略，参考之后实现了反超 cuBLAS 的性能。很有成就感，对不对！

## 7. 一些讨论

- 实际 benchmark 结果绝对值会有波动（测试设备为 RTX 5060 移动版，无法排除动态频率、桌面和系统应用的影响），本文为了行文流畅，避免前后结果矛盾，使用了同一组测试结果展示（但请放心，测试绝对值会波动，但终版 kernel 每次执行的速度都稳压 cuBLAS 一头）
- 优化过程中有一段很坎坷的经历，我花了很多时间死磕寄存器 shuffle，当时钻进牛角尖里了，一定要转置 Bs，先试了写入阶段转置，又试了直接读取 float2 后在寄存器里转置
  - 虽然最终写出来了（用了四条 __shfl_xor_sync）, 但是性能不咋地就放弃了，然后才回头逆推了 cuBLAS shared load 的策略。不过这段经历倒也加深了我对 warp shuffle 的理解
- 为什么 gemm tf32 用了 grid swizzle，而在 sgemm simt 中我就没用？其实我之前也用了，但是发现没什么效果就去掉了。应该是因为 cuda core 计算速度慢一些，计算时延已经完美隐藏访存时延了
  - 但使用 tensor core 后，算力加强，计算时延掩盖不了访存时延了，L2 的作用就凸显出来了
- 终版 kernel 已经稳定超越 cuBLAS，还有优化空间吗？当然是有的。比如，由于 mma 死板的 c fragment 输出，在写回 c 矩阵时，我还没有做到完美的事务合并。
  - 标准解法是用 As/Bs smem 作为中转 buffer，先在 smem 中排整齐，再合并写入到 global memory。但是我暂时不想写了，经历上面的过程，理解 ldmatrix/mma，以及 fragment 排布，各种映射，swizzling，我的脑子快要发烧了。读写一次 smem，又要考虑冲突问题，算了，既然性能超过 cuBLAS 就先缓缓吧~
- Callback 一下开头，为什么 cuBLAS 的 0 冲突、4 级流水线、L2 命中率也拉得很高 的 kernel 性能还没我们两级流水线的性能好？
  - 首先我认为 NV 没有专门为消费级显卡做优化，这个 kernel 不是最佳 kernel；
  - 其次，个人认为流水线是为了隐藏时延用的，并不是说越多级就越好，它为了调度 4 级流水线用了更多的寄存器资源（ncu 显示 228），只有一个 block 活跃
  - 我们的 tiling 策略使得 kernel 的计算访存比要比 cuBLAS 的更高
    - 我是用的 128x128，cuBLAS 用的 64x256，这意味着每个 block 分别需要搬运 128+128 和 64+256 单位的数据量
    - 我的计算访存比是 cuBLAS 的 (128*128 / (128+128)) / (64*256 / (64+256)) = 1.25 倍
    - 再加上两级流水线降低了寄存器压力，可以有两个活跃的 block。让 SM 有充足的 warp 进行调度，从而更好地隐藏访存的时延。
  - 如果有 nv 的专家路过，可以留言点评一下~
- 使用 mma tf32 的过程中，我一直有种别扭感，ldmatrix 没有 tf32 dtype，也用不了 .trans 转置修饰符（ldmatrix 并不关心上层的高级数据类型（无论是 FP16 还是 TF32），它操作的是 .b16 级别的数据块）
  - Ampere 引入的 tf32 tensor core 加速确实不完美，但经过这一番洗礼，后续写 hgemm 应该更顺畅了

以上部分讨论纯属个人推测，欢迎一起讨论

## 8. 结束

最后，相信经过这一轮下来，我们对 tensor core 相关指令的使用、warp 级协作、smem 的高效利用、以及如何通过 swizzling 和双 buffer 技术来优化性能都有了更深入的理解。
其实本文虽然对 mma/ldmatrix 的某些部分（比如 B fragment) 做了详细介绍，但没有完全解释所有细节，ldmatrix/mma 的 fragment A/B/C 的排布，不同精度 shape 下其实都可能不同，这些具体细节还是请参考 NVIDIA 的官方文档和 PTX 手册看看吧。

如有错误，请大家指正。完整 kernel 和测试代码可以从 github 获取，欢迎大家关注我的手撕算子系列 vitamin-cuda 项目：

以上。
