# [CUDA 优化实战] hgemm sm120 - 100KB SMEM 中的“微雕”战争：Tensor-core、TMA、ldmatrix、mma

>
> 对不起朋友们，本来我说 gemm 系列不会有后续，但我食言了，今天依然是 hgemm，不过我们要拥抱 RTX 5060 laptop 上的一切，TMA + ldmatrix + mma，挑战极限
>
> 本文适用于有一定 CUDA 编程基础，熟悉 GEMM 优化，对进阶 tensor core / 嵌入 PTX 指令 性能调优感兴趣的读者阅读
>
> 完整 kernel 和测试代码可以点击 [hgemm_sm120](/kernels/hgemm_sm120) 查看
>

## 0. 序 - sm120 - 被阉割的 blackwell

众所周知，我们 geforce 50 系列消费级显卡的 sm120 架构虽然也叫 blackwell，但和 sm100 的 B 系列 完全不是一个品种，一刀又一刀，阉割得啥都没了。tcgen5 指令没有，wgmma 没有。那有啥，有 TMA（Tensor Memory accelerator），另外 NV 还贴心的拓展了 mma 指令支持 fp8/6/4 等精度。我们今天的重点是使用 TMA，TMA 是从 hopper 架构引进专门用于加速张量数据 copy 的专用硬件，只需要一条指令即可异步的搬运一小块指定的矩阵，总之就是速度快，节省指令，异步，还自带 swizzle，配合 wgmma 简直绝了。

可惜，我们没有 wgmma。

据我所知，cutlass 都没有实现 tma + mma 这鬼畜搭配的 hgemm，仅有量化版低精度的 gemm，感兴趣参考 [cutlass example 79/87](https://github.com/NVIDIA/cutlass/tree/main/examples/79_blackwell_geforce_gemm)，所以我的这个 tma 移花接木 ldmatrix + mma.sync 实现的 hgemm，不敢说全网独一份，但肯定算稀有动物，请看官们一赏~

好，章接上文，本篇将以 M=N=K=4096（MxKxN, cuBLAS 最擅长的中等规模）的 GEMM fp16/bf16 为例，在 RTX 5060 laptop 上，使用 TMA（自带 swizzle） 移花接木 ldmatrix（手动 swizzle）+mma
实现 hgemm。并且用 cublas 和上篇文章中的 `hgemm_bcf_dbf_rw_kernel` 作为 baseline，进行对比。

这里先给结论，在我反复测试下，用上 tma 三级流水线 + 双缓冲寄存器 ldmatrix + mma 的 终版 kernel 和 我手写的纯 sm80 架构 kernel，在性能上，只能说可能具备极小的微弱的优势。在锁定最高显存和显卡频率的前提下，跑 10 次 benchmarks， 和 `hgemm_bcf_dbf_rw_kernel` 的速度对比互有胜负，大概 6/4 开

是不是有点沮丧。（其实我是有一点的，毕竟我花了很多的时间 debug 和测试，才跑通 tma 的代码）

不过换个角度想，我也了却上一篇文章的遗憾，基本实现了一个我觉得完美的 kernel，hgemm_bcf_dbf_rw_kernel 中出现的 `Uncoalesced Shared Accesses`， 这里不见了，哈哈~

只有一个 math-pipeline wait stall 提示（tensor-core 的计算速度也跟不上了） 和一个 warp Occupancy 不足提示（没有 smem 了）。其他都是完美。

本文将会给出 4 个 kernel 实现，

kernel 大纲如下（第一个是 cuBLAS kernel）

- hgemm_cublas bf16/fp16 版
- hgemm_bcf_dbf_rw bf16/fp16 版 (ldmatrix + mma, As/Bs swizzle bcf, double buffer, coalesced r/w gmem, 重构版，抽象出 copy，ldmatrix，mma compute 等函数）
- hgemm_k_stages bf16/fp16 版 （基于 hgemm_bcf_dbf_rw 改造的 kernel，可支持 3 级流水线，smem 上限了）
- hgemm_tma_r_k_stages bf16/fp16 版 （基于 hgemm_k_stages 改造的 kernel，将 cp.async 读取 gmem 替换为 TMA copy)
- hgemm_tma_rw_k_stages （这个是 todo，用 TMA copy 回 gmem，但我其实没心气做了，因为预期没有多少收益）

## 1. hgemm_bcf_dbf_rw

这个 kernel 在上一篇文章已经详细介绍过了，如何从基础版演化为终版 kernel，感兴趣的朋友请移步 []().

因此，这里只是做了一些重构工作，目的是为了让 kernel 结构更简短清晰一些，方便改造为多级流水线。
所以直接上代码：

```cpp
// a block calculate c[128][128]
template <const int BM = 128, const int BN = 128, const int BK = 32, typename T>
__global__ void hgemm_bcf_dbf_rw_kernel(T *a, T *b, T *c, int m, int n, int k) {
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

    // warp tiling
    // 每个 warp 负责  64 x 32 的 C 矩阵块
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // 寄存器总量：M 维 4 块 * N 维 4 块 * 每块 4 个寄存器 = 64
    float sum[4][4][4] = {0.f};

    T *global_a_ptr = &a[(by * BM + load_a_row) * k + load_a_col];
    T *global_b_ptr = &b[load_b_row * n + bx * BN + load_b_col];

    // ----------------------------- Prologue 先加载一次 As/Bs
    cp_async_load_A<BK>(smem.As[0], load_a_row, load_a_col, global_a_ptr, k);
    cp_async_load_B<BK, BN>(smem.Bs[0], load_b_row, load_b_col, global_b_ptr, n);

    CP_ASYNC_COMMIT_GROUP();
    cp_async_wait_group<0>();
    __syncthreads();

    int read_idx = 0;
    int write_idx = 1;

    // 主循环
    for (int bk = 32; bk < k; bk += BK) {

        // 推进指针
        global_a_ptr += BK;
        global_b_ptr += BK * n;

        // 1. cp.async load A/B
        cp_async_load_A<BK>(smem.As[write_idx], load_a_row, load_a_col, global_a_ptr, k);
        cp_async_load_B<BK, BN>(smem.Bs[write_idx], load_b_row, load_b_col, global_b_ptr, n);

        CP_ASYNC_COMMIT_GROUP();

        // 2. Tensor Core 计算阶段 (k 维度分两次，一次 16 个 k)
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 16;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // 4 次 ldmatrix A (4 * 16 = 64 行）
            ldmatrix_A<BK>(reg_a, smem.As[read_idx], warp_id_m, lane_id, k_offset);

            // 4 次 ldmatrix B (4 * 8 = 32 列）
            ldmatrix_B<BN, BK>(reg_b, smem.Bs[read_idx], warp_id_n, lane_id, k_offset);

            // MMA 核心运算：4x4 次 m16n8k16
            mma_compute<T>(sum, reg_a, reg_b);
        }

        read_idx ^= 1;
        write_idx ^= 1;

        cp_async_wait_group<0>();
        __syncthreads();
    }
    // ------------------- Epilogue 最后计算一次再写回
#pragma unroll
    for (int k_step = 0; k_step < 2; ++k_step) {
        int k_offset = k_step * 16;

        uint32_t reg_a[4][4];
        uint32_t reg_b[4][2];

        // 4 次 ldmatrix A (4 * 16 = 64 行）
        ldmatrix_A<BK>(reg_a, smem.As[read_idx], warp_id_m, lane_id, k_offset);

        // 4 次 ldmatrix B (4 * 8 = 32 列）
        ldmatrix_B<BN, BK>(reg_b, smem.Bs[read_idx], warp_id_n, lane_id, k_offset);

        // MMA 核心运算：4x4 次 m16n8k16
        mma_compute<T>(sum, reg_a, reg_b);
    }

    write_c_via_smem<BM, BN>(c, by, bx, n, sum, warp_id_m, warp_id_n, lane_id, tid, smem.Cs);
}
```

其中 cp_async_load_A，cp_async_load_B，ldmatrix_A，ldmatrix_B，mma_compute，write_c_via_smem 都改成了函数，具体不贴了，免得文章冗长，详细代码还请移步 github 查看。

## 2. hgemm_k_stages

我们稍微说明一下上一个 kernel 的 smem 的部分细节，BMxBNxBK 是 128x128x32，双 buffer 流水线，所以一共 是 128x32x2x2x2 = 32KB，如果增加一级流水线，那么就需要 128*32*2*2 = 16KB，加起来 48KB，正好是我一个 block 的所能使用的 smem 上限（总上限 100KB，单个 block 上限 48KB）。

因此我想先实现一个三级流水线 kernel，看看是否有收益。
三级流水线流程

- prologue：cp.async 预先发起加载两个 stage 的 buffer 的指令，即 commit 两个 group，wait 直到第一个 group 完成加载
- 主循环 main loop
  - 发起加载最后一个 stage 的 buffer，commit group
  - 开始 ldmaxtrix + mma 计算
  - 全局指针步进，流水线往下推进一级，同样等待最早的那个 group 加载完毕
- eplogue：
  - 等待两个 groupcopy 完成
  - 执行计算
  - 利用 smem bufer 中转写回 gmem

代码：

```c++
// 这里 As/Bs 的 Tiling 策略维持在 128x128x32，为了压进单个 Block 48KB 的 SMEM 限制，我们只能做到 3 Stage，这是 SM120 上的物理极限。
template <const int BM = 128, const int BN = 128, const int BK = 32, const int STAGES = 3, typename T>
__global__ void hgemm_k_stages_kernel(T *a, T *b, T *c, int m, int n, int k) {
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

    // A/B 都行优先，用 union 复用同一块内存，写法优雅
    __shared__ __align__(128) union {
        // 前半段计算用的 A 和 B
        struct {
            T As[STAGES][BM][BK];
            T Bs[STAGES][BK][BN];
        };
        // 后半段写回用的 C
        T Cs[BM][BN];
    } smem;

    // warp tiling
    // 每个 warp 负责  64 x 32 的 C 矩阵块
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // 寄存器总量：M 维 4 块 * N 维 4 块 * 每块 4 个寄存器 = 64
    float sum[4][4][4] = {0.f};

    T *global_a_ptr = &a[(by * BM + load_a_row) * k + load_a_col];
    T *global_b_ptr = &b[load_b_row * n + bx * BN + load_b_col];

    // 1. prologue: 加载 stages-1 个 As/Bs 块
#pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) {
        cp_async_load_A<BK>(smem.As[i], load_a_row, load_a_col, global_a_ptr, k);
        cp_async_load_B<BK, BN>(smem.Bs[i], load_b_row, load_b_col, global_b_ptr, n);

        CP_ASYNC_COMMIT_GROUP();

        global_a_ptr += BK;
        global_b_ptr += BK * n;
    }
    // commit 了两个 group, 允许 1 个 group 后台在 cp.async, 即等最早加载的 load 完毕
    cp_async_wait_group<STAGES - 2>();
    __syncthreads();

    // 状态指针初始化
    int load_stage = STAGES - 1; // 下一个要 Load 的位置
    int compute_stage = 0;       // 当前要 Compute 的位置

    // 2. main loop
    for (int bk = (STAGES - 1) * BK; bk < k; bk += BK) {

        // 1. 先发起 cp.async load As/Bs 到 load_stage
        cp_async_load_A<BK>(smem.As[load_stage], load_a_row, load_a_col, global_a_ptr, k);
        cp_async_load_B<BK, BN>(smem.Bs[load_stage], load_b_row, load_b_col, global_b_ptr, n);

        CP_ASYNC_COMMIT_GROUP();

        // 2. Tensor Core 计算阶段 (k 维度分两次，一次 16 个 k)
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 16;
            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // 4 次 ldmatrix A (4 * 16 = 64 行）
            ldmatrix_A<BK>(reg_a, smem.As[compute_stage], warp_id_m, lane_id, k_offset);

            // 4 次 ldmatrix B (4 * 8 = 32 列）
            ldmatrix_B<BN, BK>(reg_b, smem.Bs[compute_stage], warp_id_n, lane_id, k_offset);

            // MMA 核心运算：4x4 次 m16n8k16
            mma_compute<T>(sum, reg_a, reg_b);
        }

        // 推进指针
        global_a_ptr += BK;
        global_b_ptr += BK * n;

        // 流水线往下推一级
        load_stage = (load_stage + 1 == STAGES) ? 0 : load_stage + 1;
        compute_stage = (compute_stage + 1 == STAGES) ? 0 : compute_stage + 1;

        // 保障最早的 group load 好
        cp_async_wait_group<STAGES - 2>();
        __syncthreads();
    }

    // 3. epilogue 最后计算 stages-1 次 再写回
    cp_async_wait_group<0>();
    __syncthreads();
#pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) {
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 16;
            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // 4 次 ldmatrix A (4 * 16 = 64 行）
            ldmatrix_A<BK>(reg_a, smem.As[compute_stage], warp_id_m, lane_id, k_offset);

            // 4 次 ldmatrix B (4 * 8 = 32 列）
            ldmatrix_B<BN, BK>(reg_b, smem.Bs[compute_stage], warp_id_n, lane_id, k_offset);

            // MMA 核心运算：4x4 次 m16n8k16
            mma_compute<T>(sum, reg_a, reg_b);
        }

        compute_stage = (compute_stage + 1 == STAGES) ? 0 : compute_stage + 1;
    }

    write_c_via_smem<BM, BN>(c, by, bx, n, sum, warp_id_m, warp_id_n, lane_id, tid, smem.Cs);
}
```

ok，到现在我们还有大动作，所以没有什么特别可说的，下一小节见

## 3. hgemm_tma_r_k_stages_kernel

在我刚开始准备手搓 TMA copy + ldmatrix，我面临着两个问题

- 手搓 TMA copy 如何实现？
- TMA swizzle 如何能和我自己手写的 swizzle 对齐

第一个问题，经过一番坎坷的学习过程（研究 tensorMap 和 `cp.async.bulk` + mbarrier），我终于了解了整个流程，首先要在 host 端创建 CUtensorMap，用来描述 gmem 中的矩阵的 tiling size、内/外层循环的dim/stride等等，然后在 kernel 端使用 `cp.aysnc.bulk` 发起 TMA copy 请求 + mbarrier 相关命令进行同步。

### 3.1 host 端

这里给一下 host 端 tensorMap 创建方法，我们会用到 cuTensorMapEncodeTiled ，其定义和解释如下:

```cpp
CUresult CUDAAPI
cuTensorMapEncodeTiled(CUtensorMap *tensorMap, // 你要创建的 tensormap 指针
                        CUtensorMapDataType tensorDataType, //数据类型，bf16/fp16
                        cuuint32_t tensorRank, //矩阵的秩 (2D 矩阵传 2)
                        void *globalAddress, //矩阵首地址
                        const cuuint64_t *globalDim, // 全局形状。注意：必须把最内层连续维度放在第 0 位！行优先 A(MxK) 传 {K, M}
                        const cuuint64_t *globalStrides, // 步幅数组 （长度为 Rank-1)。2D 矩阵只需传 1 个元素：行跨度的字节数 （如 K * sizeof(half)
                        const cuuint32_t *boxDim, // TMA 一次搬运的 Tile 大小。维度顺序必须与 globalDim 严格对齐 （如 {BK, BM})
                        const cuuint32_t *elementStrides, //元素间跨度。对于密集矩阵，各维度全为 1，传 {1, 1}
                        CUtensorMapInterleave interleave,// 数据交错模式，一般选 NONE
                        CUtensorMapSwizzle swizzle, // swizzle 类型，可选 None，32B，64B，128B，还有几种特殊的没仔细研究
                        CUtensorMapL2promotion l2Promotion, // L2 缓存驻留粒度，推荐与缓存行对齐的 128B
                        CUtensorMapFloatOOBfill oobFill); // 越界填充策略。选 NONE 时，TMA 硬件会自动在边界外填充 0，省去繁琐的越界判断
```

硬件 TMA 最强大的地方就在于，如果你请求的 boxDim 越过了 globalDim 的边界（比如矩阵边缘 Padding），TMA 硬件会自动帮你把越界的地方塞满 0，完全不需要你在 Kernel 里写 if (x < M && y < N) 这种恶心的边界判断分支，极大地释放了 ALU 算力！写起 kernel 也极其丝滑（前提是 smem 有多余空间），我的卡就没这个福了

好，接下来我们要用这个函数创建 a/b 矩阵的 tensorMap。

问题来了，TMA 引擎在开启 128B Swizzle（这是避免 Bank Conflict 的命脉）时，有一个冷酷的硬件限制——它要求你请求的 boxDim 中，最内层的连续维度（Fastest Changing Dimension）大小必须不多不少，正好是 128 字节！

但我们的 Tiling 策略是 BN = 128。我们用的是 fp16/bf16，每个元素 2 字节，这意味着 B 矩阵一个 Tile 的行宽高达 256 字节！如果直接把 {BN, BK} 塞给 TMA，它会当场罢工，抛出 CUDA_ERROR_INVALID_VALUE。(As 的行是 BK，不论 32 还是 64 都没有超过 128B，所以没有这个问题）

怎么办？为了满足硬件的要求，我们这里用了一个小技巧：把 B 的 Tile 物理上劈成两半（Chunking），用两次 TMA 发射来完成一次逻辑上的搬运。
看代码：

```cpp
template <typename T, const int rowBytes = 128>
inline CUtensorMap
create_tensor_map(T *global_address, uint64_t fast_dim, uint64_t slow_dim, uint32_t fast_box, uint32_t slow_box) {
    CUtensorMap tmap;
    CUtensorMapDataType type =
        std::is_same_v<T, __half> ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    CUtensorMapSwizzle swizzle = rowBytes == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B;

    // TMA 的核心逻辑：第 0 维永远是内存里最连续的维度 (Fastest Changing Dimension)
    uint64_t globalDim[2] = {fast_dim, slow_dim};
    uint64_t globalStrides[1] = {fast_dim * sizeof(T)}; // 外层维度的跨度（字节）
    uint32_t boxDim[2] = {fast_box, slow_box};
    uint32_t elementStrides[2] = {1, 1};

    CUresult res = cuTensorMapEncodeTiled(&tmap,
                                          type,
                                          2, // Tensor Rank （二维矩阵）
                                          global_address,
                                          globalDim,
                                          globalStrides,
                                          boxDim,
                                          elementStrides,
                                          CU_TENSOR_MAP_INTERLEAVE_NONE,
                                          swizzle, // 对应 swizzle
                                          CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    TORCH_CHECK(res == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed!");
    return tmap;
}

// ---------------- tma func binding
#define binding_tiled_tma_func_gen(name, BK)                                                                           \
    void name##_##BK(torch::Tensor a, torch::Tensor b, torch::Tensor c) {                                              \
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
        const int smem_size = BM * BK * 2 * 3 * 2 + 24;                                                                \
        if (a.dtype() == torch::kHalf) {                                                                               \
            CUtensorMap tma_a =                                                                                        \
                create_tensor_map<__half, BK * 2>(reinterpret_cast<__half *>(a.data_ptr()), K, M, BK, BM);             \
            CUtensorMap tma_b = create_tensor_map<__half>(reinterpret_cast<__half *>(b.data_ptr()), N, K, BN / 2, BK); \
                                                                                                                       \
            cudaFuncSetAttribute(                                                                                      \
                name##_kernel<BM, BN, BK, 3, __half>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);         \
            name##_kernel<BM, BN, BK, 3><<<blocks_per_grid, threads_per_block, smem_size, stream>>>(                   \
                tma_a, tma_b, reinterpret_cast<__half *>(c.data_ptr()), M, N, K);                                      \
        } else {                                                                                                       \
            CUtensorMap tma_a = create_tensor_map<__nv_bfloat16, BK * 2>(                                              \
                reinterpret_cast<__nv_bfloat16 *>(a.data_ptr()), K, M, BK, BM);                                        \
            CUtensorMap tma_b =                                                                                        \
                create_tensor_map<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16 *>(b.data_ptr()), N, K, BN / 2, BK);   \
            cudaFuncSetAttribute(                                                                                      \
                name##_kernel<BM, BN, BK, 3, __nv_bfloat16>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);  \
            name##_kernel<BM, BN, BK, 3><<<blocks_per_grid, threads_per_block, smem_size, stream>>>(                   \
                tma_a, tma_b, reinterpret_cast<__nv_bfloat16 *>(c.data_ptr()), M, N, K);                               \
        }                                                                                                              \
    }
```

为了复用代码，我写了一个宏和一个模板函数。但我们主要看 tma_b 的创建，可以看到我传了 boxDim{BN/2,BK}进去，这样一行就是 128B 了。

### 3.2 kernel 端

那么在 kernel 内如何分配 Bs smem 呢，我使用了 Bs[2][BK][BN/2]。

```cpp
    // 使用动态共享数组
    extern __shared__ __align__(128) uint8_t smem_buf[];
    T(*As)[BM][BK] = reinterpret_cast<T(*)[BM][BK]>(smem_buf);
    T(*Bs)[2][BK][BN / 2] = reinterpret_cast<T(*)[2][BK][BN / 2]>(smem_buf + STAGES * BM * BK * sizeof(T));
    T(*Cs)[BN] = reinterpret_cast<T(*)[BN]>(smem_buf);
```

为什么要写成 Bs[2][BK][BN/2]？
从字面意义上看，它是把一块大内存物理地切成了左右两半（Chunk 0 和 Chunk 1）。

- Chunk 0 负责装载第 0 ~ 63 列。
- Chunk 1 负责装载第 64 ~ 127 列。

这样切分后，每一个 Chunk 的行宽正好是 64 个元素（即 128 字节）。我们用数组声明的方式，强行在逻辑上契合了老黄刻在硅片底层的 128B 物理边界！当然，这种切分会对后续的 TMA 写入和 ldmatrix 读取带来寻址上的麻烦，但别慌，后文我们会用地址映射来化解这个问题。

#### 3.2.1 cp.async.bulk 和 mbarrier

在全面开始 kernel 讲解之前，先了解一下我们会用到的 ptx 指令，主要是：

- mbarrier
  - 我们 3 级流水线，所以需要 3 个 mbarrier，mbarrier 是 smem 上的 8 字节变量，用于线程同步，一般 init 和 arrive 对称出现
- cp.async.bulk
  - 用于发起 TMA 拷贝，给出矩阵 tile 的左上角基地址，tma 就会开始自动异步搬运一整块 tile

具体黑魔法 ptx 代码如下，我加了一些我们这个 case 用到的详细说明：

```cpp
// MBarrier 类型定义 （硬件要求 8 字节对齐）
typedef uint64_t mbarrier_t;

// 在 smem 上初始化 64 位的屏障变量 （只需在 prologue 中由单线程调用）
// 在 TMA 模式下，真正的“数据生产者”是硬件 DMA 引擎。我们的单线程只负责下达搬运指令，mbarrier 只需要等待【TMA 硬件这 1 个实体】把数据搬完并自动打卡，所以 expected_count 设置为 1。
__device__ __forceinline__ void mbarrier_init(mbarrier_t *mbar, uint32_t expected_count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar))),
                 "r"(expected_count));
}
// 设定 TMA 传输的预期字节数，需要向指定的 mbar 汇报
// 传统同步是等“线程”arrive，这里是等“字节”arrive。硬件搬完这么多字节后，会自动触发 arrive 翻转 phase。
__device__ __forceinline__ void mbarrier_expect_tx(mbarrier_t *mbar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(
                     static_cast<uint32_t>(__cvta_generic_to_shared(mbar))),
                 "r"(tx_bytes));
}

// 计算线程消费完数据后，提交到达信号 （翻转 Phase)。我们用不到，因为没有 warp specialized，或者说所有 warp 都是消费者，直接使用 __syncthreads() 同步了
// 在 warp specialized 编程中，消费者线程会向对应 mbar 汇报数据消费完了
__device__ __forceinline__ void mbarrier_arrive(mbarrier_t *mbar) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar))));
}

// 计算线程同步等待 TMA 数据就绪 （自带休眠，不占 ALU 算力）
// 有点天书，主要逻辑是 ：
//   申请一个名为 p 的临时谓词寄存器（布尔值），用于存储 mbarrier 状态检查的结果。
//   mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2：判断 mbar 内部 phase 标志是否和给定的 phase 一致，不一致说明 tma 还没完成 copy，则挂起 ticks 周期，让渡出 ALU 给其他 warp 调度
//   @p bra DONE：如果 p 为 true，即说明 mbarrier phase 已经反转，tma 完成 copy，直接跳跃到 DONE 标签之后的指令
//   bra LAB_WAIT: 如果走到这里，说明是那一千万个时钟周期超时了（极小概率），那就跳回 LAB_WAIT 继续休眠。
//   最后的"memory" 是编译器内存屏障（破坏性描述符）。它防止编译器过度优化读到脏数据，作用是提示编译器：TMA 硬件刚刚在后台偷偷篡改了共享内存（As, Bs 矩阵），该指令之后所有寄存器缓存的共享内存变量必须强制失效，重新读取。
_device__ __forceinline__ void mbarrier_wait(uint64_t* mbar, uint32_t phase) {
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    // 设定一个极大的挂起超时周期（0x989680 = 10,000,000 个时钟周期）
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred p; \n\t"
        "LAB_WAIT: \n\t"
        // 注意这里的第三个参数 %2
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2; \n\t"
        "@p bra DONE; \n\t"
        "bra LAB_WAIT; \n\t"
        "DONE: \n\t"
        "}\n"
        :
        : "r"(mbar_addr), "r"(phase), "r"(ticks)
        : "memory"
    );
}

// global to shared::cta 2d TMA 搬运
__device__ __forceinline__ void cp_async_bulk_tensor_2d(
    mbarrier_t *mbar, const void *tmap, const void *smem_ptr, int32_t fast_coord, int32_t slow_coord) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));

    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3}], [%4];\n" ::"r"(smem_addr),
                 "l"(tmap),
                 "r"(fast_coord),
                 "r"(slow_coord),
                 "r"(mbar_addr)
                 : "memory");
}
```

所以 kernel 的主要 TMA 相关逻辑就是（在我们这个非 warp 特化的开发模式下，让线程 0 负责 TMA 调度）

- copy 线程（线程 0）：
  - 初始化 3 个 stage 的 mbarrier 变量，
  - 并向 mbarrier 汇报一个 stage 要搬运的数据量
  - 使用 tensorMap，发起 TMA copy 请求（cp.async.bulk）
- 计算线程（所有线程）：
  - 共同轮询等待（mbarrier_wait），对应 stage 的 mbarrier 翻转为指定 phase 时，线程全都被唤醒
  - 开始 ldmatrix+mma 计算

### 3.3 kernel 设计

知道大概流程后，可以着手设计 kernel 了，其实上面还有一个问题我还没回复，就是 TMA 的 swizzle 和我的手写 swizzle 如何对齐，我没有花精力找到相关资料确定 TMA 硬件 swizzle 方式，所以只是猜，但猜就有概率问题。

为了减少测试量。我做出了一个违背“祖宗”的决定。我放弃了 BMxBNxBK=128x128x32 的 tiling，而是将 BK 设置为 64！这样 a 矩阵的一个 tile 一行也是 128B，b 矩阵经过我们劈开一个 chunk 的一行也是 128B，只要对齐 128B 的 TMA 和 ldmatrix swizzle，就能跑通了。

代价就是我需要开 128*64*2*3*2 = 96 KB 的 smem，只能驻留一个 block，还要用 cuda 魔法 cudaFuncSetAttribute + 动态 smem 数组。

此外，由于 TMA 只需要一条指令，解放了 unroll 的所有地址变量寄存器，所以我激进地加入了 双 buffer 寄存器读取 As/Bs（虽然作用不大）

这里给出完整 kernel 代码：

```cpp
template <const int BK, typename T>
__device__ __forceinline__ void
ldmatrix_A_tma(uint32_t reg_a[4][4], T (*As)[BK], int warp_id_m, int lane_id, int k_offset) {

    // 4 次 ldmatrix A (4 * 16 = 64 行）
#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
        int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
        int a_col = k_offset + (lane_id / 16) * 8;
        if constexpr (BK == 32) {
            uint32_t smem_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&As[a_row][SWIZZLE_64B_TMA(a_row, a_col)]));
            LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
        } else {
            uint32_t smem_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&As[a_row][SWIZZLE_128B_TMA(a_row, a_col)]));
            LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
        }
    }
}

template <const int BN, const int BK, typename T>
__device__ __forceinline__ void
ldmatrix_B_tma(uint32_t reg_b[4][2], T (*Bs)[BK][BN / 2], int warp_id_n, int lane_id, int k_offset) {
#pragma unroll
    for (int n_idx = 0; n_idx < 4; ++n_idx) {
        int b_row = k_offset + (lane_id % 16);
        int b_col = warp_id_n * 32 + n_idx * 8;

        // 这里要区分 chunk
        int chunk_idx = b_col / (BN / 2);
        int local_col = b_col % (BN / 2);

        uint32_t smem_addr =
            static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[chunk_idx][b_row][SWIZZLE_128B_TMA(b_row, local_col)]));
        LDMATRIX_X2_TRANS(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
    }
}
// -------------------   tma r + mma -------------------
// a block calculate c[128][128]
template <const int BM = 128, const int BN = 128, const int BK = 64, const int STAGES = 3, typename T>
__global__ void hgemm_tma_r_k_stages_kernel(
    __grid_constant__ const CUtensorMap tma_a, __grid_constant__ const CUtensorMap tma_b, T *c, int m, int n, int k) {
    // grid swizzling
    int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8; // 将执行块设置为 8 的宽度

    int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    int by = (linear_id / SWIZZLE_W) % gridDim.y;

    int tid = threadIdx.x; // 0~255
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // 使用动态共享数组
    extern __shared__ __align__(128) uint8_t smem_buf[];
    T(*As)[BM][BK] = reinterpret_cast<T(*)[BM][BK]>(smem_buf);
    T(*Bs)[2][BK][BN / 2] = reinterpret_cast<T(*)[2][BK][BN / 2]>(smem_buf + STAGES * BM * BK * sizeof(T));
    T(*Cs)[BN] = reinterpret_cast<T(*)[BN]>(smem_buf);
    // 把 mbar 放在末尾 ( 8 字节对齐，3 个 stages)
    mbarrier_t *mbar = reinterpret_cast<mbarrier_t *>(smem_buf + BM * BK * sizeof(T) * STAGES * 2);

    // 初始化 MBarrier （仅需 tid 0 执行，期待到达次数为 1，因为只有 TMA 会给它发信号）
    if (tid == 0) {
        for (int i = 0; i < STAGES; ++i)
            mbarrier_init(&mbar[i], 1);
    }
    __syncthreads(); // 保证 MBarrier 初始化完毕

    // warp tiling
    // 每个 warp 负责  64 x 32 的 C 矩阵块
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // 寄存器总量：M 维 4 块 * N 维 4 块 * 每块 4 个寄存器 = 64
    float sum[4][4][4] = {0.f};

    // 每次 TMA 需要搬运的总字节数
    const uint32_t tx_bytes = (BM * BK + BK * BN) * sizeof(T);

    // 只保留一个极其简单的坐标跟踪变量 （因为 TMA 的 Host 描述符里已经知道了跨度）
    int load_k_coord = 0;

    // 1. prologue 加载 STAGES - 1 块
    for (int i = 0; i < STAGES - 1; ++i) {
        if (tid == 0) {
            // 设定这个 mbarrier 需要等多少字节的数据落盘
            mbarrier_expect_tx(&mbar[i], tx_bytes);

            cp_async_bulk_tensor_2d(&mbar[i], &tma_a, As[i], load_k_coord, by * BM);
            // Bs 要 copy 两次，分成两个 chunk
            cp_async_bulk_tensor_2d(&mbar[i], &tma_b, Bs[i][0], bx * BN, load_k_coord);
            cp_async_bulk_tensor_2d(&mbar[i], &tma_b, Bs[i][1], bx * BN + BN / 2, load_k_coord);
        }
        load_k_coord += BK;
    }

    int load_stage = STAGES - 1;
    int compute_stage = 0;
    int wait_phase = 0; // MBarrier 天然的 0/1 交替相位开关
    int total_k_step = BK / 16; // 根据 BK 自适应 step
    // 2. main loop
    for (int bk = (STAGES - 1) * BK; bk < k; bk += BK) {

        // 发起下一轮的 TMA （依然只有 tid 0 干活）
        if (tid == 0) {
            mbarrier_expect_tx(&mbar[load_stage], tx_bytes);
            cp_async_bulk_tensor_2d(&mbar[load_stage], &tma_a, As[load_stage], load_k_coord, by * BM);
            cp_async_bulk_tensor_2d(&mbar[load_stage], &tma_b, Bs[load_stage][0], bx * BN, load_k_coord);
            cp_async_bulk_tensor_2d(&mbar[load_stage], &tma_b, Bs[load_stage][1], bx * BN + BN / 2, load_k_coord);
        }
        load_k_coord += BK;

        // 所有线程：轮询等待当前 compute_stage 的数据被 TMA 搬运完毕
        mbarrier_wait(&mbar[compute_stage], wait_phase);

        // 寄存器双 buffer: ldmatrix + mma
        uint32_t reg_a[2][4][4], reg_b[2][4][2];
        ldmatrix_A_tma<BK>(reg_a[0], As[compute_stage], warp_id_m, lane_id, 0);
        ldmatrix_B_tma<BN, BK>(reg_b[0], Bs[compute_stage], warp_id_n, lane_id, 0);
        int read_idx = 0, write_idx = 1;
#pragma unroll
        for (int k_step = 0; k_step < total_k_step; ++k_step) {
            if (k_step < total_k_step - 1) {
                int next_k_offset = (k_step + 1) * 16;
                ldmatrix_A_tma<BK>(reg_a[write_idx], As[compute_stage], warp_id_m, lane_id, next_k_offset);
                ldmatrix_B_tma<BN, BK>(reg_b[write_idx], Bs[compute_stage], warp_id_n, lane_id, next_k_offset);
            }
            mma_compute<T>(sum, reg_a[read_idx], reg_b[read_idx]);
            read_idx ^= 1;
            write_idx ^= 1;
        }

        // 直接同步，没有 warp 特化，不需要 arrive
        __syncthreads();

        // 状态轮转
        load_stage = (load_stage + 1 == STAGES) ? 0 : load_stage + 1;
        compute_stage = (compute_stage + 1 == STAGES) ? 0 : compute_stage + 1;

        // 完成一次三级流水线，三个 mbarrier 的 phase 都反转了，我们也要反转给定的 wait_phase
        if (compute_stage == 0)
            wait_phase ^= 1;
    }
    // 3. epilogue 计算 stages-1 次
#pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) {
        // 继续等 TMA
        mbarrier_wait(&mbar[compute_stage], wait_phase);

        // 寄存器双 buffer
        uint32_t reg_a[2][4][4], reg_b[2][4][2];
        ldmatrix_A_tma<BK>(reg_a[0], As[compute_stage], warp_id_m, lane_id, 0);
        ldmatrix_B_tma<BN, BK>(reg_b[0], Bs[compute_stage], warp_id_n, lane_id, 0);
        int read_idx = 0, write_idx = 1;
#pragma unroll
        for (int k_step = 0; k_step < total_k_step; ++k_step) {
            if (k_step < total_k_step - 1) {
                int next_k_offset = (k_step + 1) * 16;
                ldmatrix_A_tma<BK>(reg_a[write_idx], As[compute_stage], warp_id_m, lane_id, next_k_offset);
                ldmatrix_B_tma<BN, BK>(reg_b[write_idx], Bs[compute_stage], warp_id_n, lane_id, next_k_offset);
            }
            mma_compute<T>(sum, reg_a[read_idx], reg_b[read_idx]);
            read_idx ^= 1;
            write_idx ^= 1;
        }

        compute_stage = (compute_stage + 1 == STAGES) ? 0 : compute_stage + 1;
        if (compute_stage == 0)
            wait_phase ^= 1;
    }

    // 4. 写回
    write_c_via_smem<BM, BN>(c, by, bx, n, sum, warp_id_m, warp_id_n, lane_id, tid, Cs);
}
```

ldmatrix Bs 的时候，要先区分一下 chunk，还好我们只是完整的切成两大块，并不复杂。其他的就没有什么了，代码中也基本有注释帮助解释。

此外，细心的朋友可能还注意到我的 kernel 传入 tensorMap 用了`__grid_constant__`, 这个约束关键字是让 gpu 强制传入这个变量到常量存储。

#### 3.3.1 踩坑血泪史：不可忽视的 `__grid_constant__`

事实上一开始，我在 tensorMap 的构建上就卡住了很久。我遇到了一个诡异的问题：一跑就挂。经过痛苦的排查，我发现是因为 tensorMap 本质上是一个占据 128 字节的复杂结构体（Descriptor）。按照 CUDA 的默认传参规则，这么大的结构体作为参数传入时，很容易被编译器分配到 Local Memory 中。而 TMA 硬件引擎要求 tensorMap 必须存在于全局常量区。

解决办法就是在声明 Kernel 的参数时，对 tensorMap 变量 加上 `__grid_constant__` 关键字。这个修饰符，会强制编译器将 TensorMap 存放在 Constant Memory 中，并保证其在整个 Grid 生命周期的可见性与只读性。

初玩 TMA 的朋友，这个坑务必警醒！

## 4. 殊途同归：TMA 与 ldmatrix 的 Swizzle 握手

总之，经过一番痛苦的 debug 过程，我终于跑通了上面那个 kernel, 也通过了 diff_check，说明 TMA 128B swizzle 和我之前对 Bs 进行的 swizzle 一致 ：`((col) ^ (((row) & 0x7) << 3))`
所以说啊，gpu 设计和进化也要讲基本法，我手动 ldmatrix 的 swizzle 既然和 TMA 的一样，说明底层逻辑其实一样的。

我就猜想 64B 的 swizzle 是不是也一样呢。

于是我又对代码做了一点调整，兼容 BK=32/64 两种情况，ldmatrix As 就用我原来的 swizzle_a。然后设置 BK=32，这样 smem 用量减半，又可以有两个 block 活跃。

改完后，一跑就通过 diff check 了。哈哈，果然皇天不负有心人。TMA 的黑盒 swizzle 逻辑我不知道，但在严格精度 diff 测试下，我的手动 Swizzle 逻辑与 TMA 硬件底层的黑盒行为完美契合，误差为 0。

这是纯粹的逆向工程浪漫——既然硬件不说，我们就用结果去反推逻辑。

## 5.benchmark、ncu report 和分析

本次测试，我预期终版 kernel 会是一个和 baseline 不相上下的结果，所以我进行了相对严格的对比，尽量关闭所有程序+显存显卡锁定频率，然后在低温环境进行测试。

直接贴 benchmark 结果：

```yaml
####################################################################################################
n: 4096, m: 4096, k: 4096
torch                                    mean time: 4.011336 ms, 34.26 tflops
hgemm_cublas                             mean time: 4.258727 ms, speedup: 0.94, tflops: 32.27
hgemm_bcf_dbf_rw                         mean time: 4.042202 ms, speedup: 0.99, tflops: 34.00
hgemm_k_stages                           mean time: 4.131343 ms, speedup: 0.97, tflops: 33.27
hgemm_tma_r_k_stages_64                  mean time: 4.287896 ms, speedup: 0.94, tflops: 32.05
hgemm_tma_r_k_stages_32                  mean time: 4.005909 ms, speedup: 1.00, tflops: 34.31
```

我承认，这是我挑了一组终版 kernel 胜利的数据，毕竟花了这么多精力，还放个不如 `hgemm_bcf_dbf_rw` （实际胜负比 6/4 开吧），多煞风景。
不过在我心里，`hgemm_tma_r_k_stages_32` 和 `hgemm_bcf_dbf_rw` 就是相同等级的水平，都已经把 tensor-core 压榨到极限了（虽然前者要复杂得多）

ncu report
![p](static/hgemm_sm120_0.png)

这是我最满意的一份 ncu report，虽然终版 kernel 性能不能稳压`hgemm_bcf_dbf_rw`一头，但是 ncu summary 很干净，没有讨厌的 `Uncoalesced Shared Accesses`，只有 tensor-core 算力不足，和 register/smem 限制导致的 Occupancy 提示。

- `hgemm_tma_r_k_stages_64` 0 bank conflict
![p](static/hgemm_sm120_1.png)
- `hgemm_tma_r_k_stages_32` 也是 0 bank conflict(ncu 还是报了一点点 bank conflict，毕竟双 block 活跃，有一些 warp 间的冲突，但 swizzle 不背锅）
![p](static/hgemm_sm120_2.png)

### 一些讨论

- 为什么单纯的 2 级流水改为 3 级流水线性能没提高反而下降了
  - 流水线实质就是空间换时间，我推测是 2 级流水线寄存器压力已经很大了，3 级理论要更多的寄存器同时还要保障 2 个 block 活跃，必定重排了指令，牺牲了更多的指令级并行度
- TMA 专用硬件搬运数据确实很有用，尤其是降低寄存器用量，从我的 ncu 报告能看出
  - 128x128x32 的 tiling 策略下，即使 3 级流水线，寄存器用量已经从 `hgemm_bcf_dbf_rw` 的 128 降低到了 96，这还是我用了双寄存器缓冲的情况下啊！
  - 这意味着在 SM120 上，瓶颈已经彻底从‘访存调度’转移到了‘算力吞吐’。TMA 已经做到了它能做的一切，剩下的差距是老黄割掉的那几刀算力。
- 但光有 TMA，没有大 smem 和足够的 tensor-core 算力，也没用呀
  - ncu report 提示 `Math Pipe Throttle Stalls`, tensor core 都已经算不过来了
  - 再给我 46KB 的 smem，再加些算力，我能再多一个 block，算力吞吐能提升 50%。只能说老黄刀法好啊
- 终版 kernel 总结
  - sm120 的 tma + ldmatrix（移花接木 swizzle、双寄存器 buffer） + mma + 3 级流水线 + grid swizzle kernel
  - 我几乎用上了我能理解的一切

## 6. 结束

经过艰苦的 coding，我完成了一个自认完美的 kernel，这张卡已经被我榨干了。

这真是 gemm 系列最后一篇了，原本还有个 todo，使用 tma 将 Cs 写回 gmem，但预期收益不大，就没动力了。

所以，gemm 系列完结 ~ 撒花 ~

以上。

如有错误，请大家指正。完整 kernel 和测试代码可以从 github 获取：
