# [CUDA 优化实战] fmha sm120 : 超越 torch.sdpa(flash-attention-2)

## 0. 序 - LLM 时代高性能计算算子中的桂冠

> 本文适用于有一定 CUDA 编程基础，熟悉 GEMM/multi-head-attention 优化，对进阶 tensor core / 嵌入 PTX 指令 性能调优感兴趣的读者阅读
>
> 实际上阅读本文前最好先阅读本人先发布的 hgemm sm120 、safe online softmax 以及 gemm 系列文章。因为可能有重合的知识点，本人就一笔带过了，而朋友们还云里雾里。
>
> 完整 kernel 和测试代码可以点击 [flash_attn](/kernels/flash_attn) 查看，欢迎大家关注我的 vitamin-cuda 项目：
>

LLM 时代，几乎所有模型都有一个不可或缺的组件 Multi-Head-Attention，其计算复杂度随着序列长度成平方正比，因此常常是计算开销的大头。尽管如今已经提出了各种各样的 linear attention 及其变体，但模型效果上依然达不到 full attention 的精度。因此更折中的办法是开始混合 layer（hybrid attention），当然这和本篇文章无关。

本篇文章主要内容很纯粹，就是介绍一下如何在 5060 laptop（sm120 架构） 上不依赖第三方库，没有 cutlass，不用 cute，纯手搓一个高性能的 fmha 算子（用上 TMA+ldmatrix+mma，毕竟这就是我们的全部了）。当然了，性能超过 FA2 不是说我比 Tri dao 大神牛逼，别人的 kernel 是 sm80 架构实现，没有咱们的 TMA，寄存器压力拉满了。

实际上 flash attention 的 kernel，用 NCU profile 出来的数据还是很漂亮的。如果 Tri Dao 来写我这个 tma + ldmatrix + mma kernel，肯定稳压我一头。

算子实现的主要思想和 flash-attention-2 大体相同（虽然我没有具体去翻 FA2 的代码，cutlass、cute 什么的，真的叫人看得头痛），但论文我还是粗略翻过一下的。论文算法原理细节咱就不扣了，这个感兴趣的直接去读论文即可，我在这解读论文可就拾人牙慧了。

本文的 kernel 大纲如下：

- fmha_tma_128 （BMxBN=64x64，TMA + ldmatrix + mma）

对，没了，本文就一个算子，支持了 head_dim 为 128 （也是现在模型很常见的大小了）情况下的 causal attention。

最初我是有考虑先实现一个三矩阵连乘算子，再加入 online softmax 实现 attention，作为演化路径帮助理解。但写的时候发现尽管我抽出了大量函数出去，三矩阵连乘也已经够复杂了，索性就一个 kernel 到位。

## 1. flash attention

flash attention 是我很佩服的一个作品，众所周知，它算是第一个把 attention 中 online softmax 思想工程化落地并造成出圈影响力的工作。从 FA1 的利用 smem 做 online softmax，到 FA2 的并行策略和访问模式优化（前向计算中将内循环的规约从 V 转移到 Q，以及 sequence 并行），工作都很实在且效果明显。总之 flash attention 是很 solid 的论文作品。后面的 FA3、FA4 都是结合新显卡架构的实现，咱就不多提了。

先讲讲 MHA 本身。
首先 MHA = softmax(q @ k.T*scale)@v, 其中 scale 一半为 k 的 head_dim 的平方根，相当于对 K 做一定程度上的归一化。
正常 naive 的实现肯定会

- `s = q@k.T*scale`
- `p = softmax(s)`
- `o = p@v`

这样分步计算，会产生 q 和 k 一次读取 gmem，s 一次写入+读取 gmem，p 和 v 的 gmem 读取 ，最后 o 的写入 gmem。gpu 在计算和 gmem 之间来回奔波搬运数据。

再讲讲 flash attention 的思路。

flash attention 的思想就是把中间 s 和 p 的 gmem 请求都抹掉，利用分块（分治）的思想在 smem 上 online 缩放完成 softmax 过程，最后直接得到 o（即最终输出）。也就是，仅有 qkv 的一次gmem读，加 o 的一次gmem写。

更具体的，为了方便理解，这里我们假设 qkv 的 shape 一致（当然我的算子实现也支持 GQA），都为 [batch_size, seq_len, heads, head_dim]
同时 attention 是输出是对应 q 的，不会改变 shape，因此 o 的 shape 和 q 一致。然后整个算法流程就是

- 加载一个 q tile（下面会解释 tiling 策略）
- 沿着 seq 的方向做 loop，循环中每次加载 kv 的一个 tile
- 每次加载 kv 的 tile 时，先计算 `q@k.T`
  - 再计算 s，以及 p，这里会涉及 online softmax 的动态缩放
  - 计算 pv，累加进o
- loop 结束就得到一个最终结果的 o tile
- o 写回 gmem
- 结束

先说明我的 data tiling 策略，最终测试其实我是调了很多次，然后验算出来的一个配置（写算子本来就是这样一个脏活，没有谁一眼就能看出最佳写法）：

- 首先受限于 online softmax 的数学本质，计算 attn score 依赖完整的向量点积，因此内层 head_dim 我肯定是要一次性加载进来的。不会在 head_dim=128 上做切分
- seq_len，也就是 token 长度，这个是波动范围巨大的，所以肯定得切
- heads 数量一般就是几十到百级别，因此可以直接丢到 grid 的某个维度
- batch size 同上（实际上 prefill 一般不考虑 batchsize，或者说 batchsize 和 seq_len 合起来考虑，毕竟 seq_len 维度带来的访存压力呈平方级增长，batch_size 的影响则相对次要）

那么沿着 seq_len 一次要切多大呢，这里要考虑我的 smem 限制（总量 100KB，单个 block 上限 48KB）。经过验算我发现选择 BM=64 切 q 的 seq_len，选 BN=64 切 kv 的 seq 维度最佳。
验算一下：

- (BM*head_dim + BN*head_dim*2)*2 bytes = (64*128 + 64*128*2)*2 = 48KB（正好是我的单个 block 上限），当然由于我们要使用 TMA，还需要几个 mbarrier 变量，所以多些字节
- 这样正好可以允许我有两个 block 活跃，Occupancy 有一定保障
- 尽管这个配置下没办法考虑双 buffer 或多级流水线之类的了（咱就这点 smem，别奢求了）

那么在确定了 data tiling 之后。thread block 也就确定了。block_size=128，因为计算 `q@k.T` 是 `MxNxK = 64x64x128`, `mma` 指令用 `m16n8k16`。没有太多的 m 和 n 需要大量 warp hold。使用 4 个 warp 即可，每个 warp 正好负责 `16x64 的 MxN`，也能对应 `m16n8k16` 的指令需求（64/8=8，128/16=8，都能整除）

ok，这样过一遍，我们的 kernel launch 代码已经比较清晰了。thread_block 指定为 128，batchsize、heads、BM 序列维度 分别放在 grid 的三个维度上。

此外，这里要考虑拉满 L2 的使用率，所以我们需要相邻的 block 在 kvloop 中尽可能读取相同的 kv tile，那把序列维度放在 x 维度上，batchsize 放在 y 上，heads 放在 z。
block 是沿着 xyz 的顺序发射的，而我们是沿着 seq 维度进行 kv loop。因此，要把不会引起 kv tile 变化的维度放在内层，head 维度直接影响了是否读取不同的 kv，所以放在最外层。

可能说的有点抽象，直接看代码：

```cpp
template <typename T, const int rowBytes = 128>
inline CUtensorMap create_4d_tensor_map(T *global_address,
                                        uint64_t dim_d,
                                        uint64_t dim_h,
                                        uint64_t dim_s,
                                        uint64_t dim_b,
                                        uint64_t stride_h,
                                        uint64_t stride_s,
                                        uint64_t stride_b, // 字节跨步
                                        uint32_t box_d,
                                        uint32_t box_s) // 每次加载 (box_s x box_d) tile
{
    CUtensorMap tmap;
    CUtensorMapDataType type =
        std::is_same_v<T, __half> ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    CUtensorMapSwizzle swizzle = rowBytes == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B;

    // TMA 维度：从最内层到最外层
    uint64_t globalDim[4] = {dim_d, dim_h, dim_s, dim_b};

    // 全局各维度上 stride，字节数
    uint64_t globalStrides[3] = {stride_h, stride_s, stride_b};

    // 每个维度上加载的 tile 大小
    uint32_t boxDim[4] = {box_d, 1, box_s, 1};
    uint32_t elementStrides[4] = {1, 1, 1, 1};

    CUresult res = cuTensorMapEncodeTiled(&tmap,
                                          type,
                                          4, // 4d 矩阵
                                          global_address,
                                          globalDim,
                                          globalStrides,
                                          boxDim,
                                          elementStrides,
                                          CU_TENSOR_MAP_INTERLEAVE_NONE,
                                          swizzle,
                                          CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    TORCH_CHECK(res == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed for 4D Tensor!");
    return tmap;
}

#define binding_tiled_tma_func_gen(name, HEAD_DIM)                                                                     \
    void name##_##HEAD_DIM(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, float scale) {          \
                                                                                                                       \
        CHECK_T(q);                                                                                                    \
        CHECK_T(k);                                                                                                    \
        CHECK_T(v);                                                                                                    \
        CHECK_T(o);                                                                                                    \
                                                                                                                       \
        const int batch_size = q.size(0);                                                                              \
        const int q_len = q.size(1);                                                                                   \
        const int q_head = q.size(2);                                                                                  \
        const int head_dim = q.size(3);                                                                                \
        const int kv_len = k.size(1);                                                                                  \
        const int kv_head = k.size(2);                                                                                 \
                                                                                                                       \
        TORCH_CHECK(head_dim == HEAD_DIM, "Head dim mismatch: expected ", HEAD_DIM);                                   \
                                                                                                                       \
        int elem_bytes = q.element_size();                                                                             \
        uint64_t q_stride_h = q.stride(2) * elem_bytes;                                                                \
        uint64_t q_stride_s = q.stride(1) * elem_bytes;                                                                \
        uint64_t q_stride_b = q.stride(0) * elem_bytes;                                                                \
                                                                                                                       \
        uint64_t k_stride_h = k.stride(2) * elem_bytes;                                                                \
        uint64_t k_stride_s = k.stride(1) * elem_bytes;                                                                \
        uint64_t k_stride_b = k.stride(0) * elem_bytes;                                                                \
                                                                                                                       \
        uint64_t v_stride_h = v.stride(2) * elem_bytes;                                                                \
        uint64_t v_stride_s = v.stride(1) * elem_bytes;                                                                \
        uint64_t v_stride_b = v.stride(0) * elem_bytes;                                                                \
                                                                                                                       \
        const int BM = 64;                                                                                             \
        const int BN = 64;                                                                                             \
                                                                                                                       \
        CUtensorMap tma_q = create_4d_tensor_map<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16 *>(q.data_ptr()),       \
                                                                head_dim,                                              \
                                                                q_head,                                                \
                                                                q_len,                                                 \
                                                                batch_size,                                            \
                                                                q_stride_h,                                            \
                                                                q_stride_s,                                            \
                                                                q_stride_b,                                            \
                                                                head_dim / 2,                                          \
                                                                BM);                                                   \
                                                                                                                       \
        CUtensorMap tma_k = create_4d_tensor_map<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16 *>(k.data_ptr()),       \
                                                                head_dim,                                              \
                                                                kv_head,                                               \
                                                                kv_len,                                                \
                                                                batch_size,                                            \
                                                                k_stride_h,                                            \
                                                                k_stride_s,                                            \
                                                                k_stride_b,                                            \
                                                                head_dim / 2,                                          \
                                                                BN);                                                   \
                                                                                                                       \
        CUtensorMap tma_v = create_4d_tensor_map<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16 *>(v.data_ptr()),       \
                                                                head_dim,                                              \
                                                                kv_head,                                               \
                                                                kv_len,                                                \
                                                                batch_size,                                            \
                                                                v_stride_h,                                            \
                                                                v_stride_s,                                            \
                                                                v_stride_b,                                            \
                                                                head_dim / 2,                                          \
                                                                BN);                                                   \
                                                                                                                       \
        /* q_seq 放最内层尽量复用 L2 读取 kv tile */                                                      \
        const dim3 blocks_per_grid((q_len + BM - 1) / BM, batch_size, q_head);                                         \
        const int THREADS_PER_BLOCK = 128;                                                                             \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
        const int smem_size = (BM * HEAD_DIM + BN * HEAD_DIM * 2) * sizeof(__nv_bfloat16) + sizeof(mbarrier_t) * 3;    \
        cudaFuncSetAttribute(name##_kernel<BM, BN, HEAD_DIM, THREADS_PER_BLOCK, __nv_bfloat16>,                        \
                             cudaFuncAttributeMaxDynamicSharedMemorySize,                                              \
                             smem_size);                                                                               \
        /* launch kernel */                                                                                            \
        name##_kernel<BM, BN, HEAD_DIM, THREADS_PER_BLOCK, __nv_bfloat16>                                              \
            <<<blocks_per_grid, THREADS_PER_BLOCK, smem_size, stream>>>(                                               \
                tma_q,                                                                                                 \
                tma_k,                                                                                                 \
                tma_v,                                                                                                 \
                reinterpret_cast<__nv_bfloat16 *>(o.data_ptr()),                                                       \
                q_len,                                                                                                 \
                kv_len,                                                                                                \
                q_head,                                                                                                \
                kv_head,                                                                                               \
                scale);                                                                                                \
    }
```

tensorMap 的创建在上篇文章详细介绍过，这里依然要用到上文章的技巧，即把 head_dim 劈开两半存在 smem 上，不然就超过 TMA swizzle 的最大字节数限制（128B）了。
同样的，因为超出了几个 mbarrier 变量的用量，需要使用魔法 cudaFuncSetAttribute + 动态共享内存。

## 2. fmha 实现

上面已经讲清楚 data tiling 和 thread tiling 了。因此这里讲讲解 kernel 内的细节。

整体流程：

- 初始化 Qs、Ks、Vs 三块 smem 和三个 tma mbarrier
- 初始化 acc_o 寄存器（这是最终输出）
- 加载 Qs，wait 直到加载完成
- kv loop：沿着序列维度循环
  - 先发起加载 Ks 的 TMA 指令，再发起加载 Vs 的 TMA 指令
  - wait Ks 的 mbarrier phase 反转，然后立刻进行 Qs 和 Ks 的 mma 计算得到 acc_s（这里和 Vs 的 TMA 的拷贝是同时进行的，相当于隐藏了 Vs 的加载时延）
  - acc_s 的 online softmax 流程得到 Ps，
  - Ps 重排一下寄存器，构造成 Ps@Vs 的 mma Fragment A 的寄存器结构
  - mma 计算 Ps@Vs 得到 O
  - 同步，进入下一循环
- 此时 acc_o 寄存器就是最终输出啦，利用上篇文章的 smem 中转写回技巧，写入 gmem
- 结束

上代码：

```cpp
// tma copy implement softmax(q @ k.T*scale) @ v
template <const int BM = 64, const int BN = 64, const int HEAD_DIM = 128, const int THREADS_PER_BLOCK = 128, typename T>
__global__ void fmha_tma_kernel(const __grid_constant__ CUtensorMap tma_q,
                                const __grid_constant__ CUtensorMap tma_k,
                                const __grid_constant__ CUtensorMap tma_v,
                                T *o,
                                int q_len,
                                int kv_len,
                                int q_head,
                                int kv_head,
                                float scale) {
    // 1. 48KB smem + 3 mbarriers
    extern __shared__ __align__(128) uint8_t smem_buf[];
    T(*Qs)[BM][HEAD_DIM / 2] = reinterpret_cast<T(*)[BM][HEAD_DIM / 2]>(smem_buf); // BM*HEAD_DIM
    T(*Ks)
    [BN][HEAD_DIM / 2] = reinterpret_cast<T(*)[BN][HEAD_DIM / 2]>(smem_buf + BM * HEAD_DIM * sizeof(T)); // BN*HEAD_DIM
    T(*Vs)
    [BN][HEAD_DIM / 2] =
        reinterpret_cast<T(*)[BN][HEAD_DIM / 2]>(smem_buf + (BM + BN) * HEAD_DIM * sizeof(T)); // BN*HEAD_DIM
    T(*Os)
    [HEAD_DIM] =
        reinterpret_cast<T(*)[HEAD_DIM]>(smem_buf); // 16KB, reused at the end for writing back to global memory

    // mbar 变量放在 smem 末尾，8 字节对齐
    mbarrier_t *mbar_q = reinterpret_cast<mbarrier_t *>(smem_buf + (BM + BN * 2) * HEAD_DIM * sizeof(T));
    mbarrier_t *mbar_k =
        reinterpret_cast<mbarrier_t *>(smem_buf + (BM + BN * 2) * HEAD_DIM * sizeof(T) + sizeof(mbarrier_t));
    mbarrier_t *mbar_v = mbar_k + 1;

    // 坐标映射
    const int tid = threadIdx.x;
    const int q_tile_idx = blockIdx.x;
    const int batch_id = blockIdx.y;
    const int head_id = blockIdx.z;

    const int group_size = q_head / kv_head;
    const int kv_head_id = head_id / group_size;
    const int q_start_idx = q_tile_idx * BM;

    if (tid == 0) {
        mbarrier_init(mbar_q, 1);
        mbarrier_init(mbar_k, 1);
        mbarrier_init(mbar_v, 1);

        mbarrier_expect_tx(mbar_q, BM * HEAD_DIM * sizeof(T));
        // 发起两次加载 Qs 的 TMA 指令，左右各一半 128B
        cp_async_bulk_tensor_4d(mbar_q, &tma_q, Qs[0], 0, head_id, q_start_idx, batch_id);
        cp_async_bulk_tensor_4d(mbar_q, &tma_q, Qs[1], 64, head_id, q_start_idx, batch_id);
    }
    __syncthreads();
    mbarrier_wait(mbar_q, 0); // Wait for Q to finish loading (Q is used throughout the inner loop)

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warp_row_offset = warp_id * 16;

    // 4. 加载 Qs
    uint32_t reg_q[8][4];
    ldmatrix_Qs<BM, HEAD_DIM>(reg_q, Qs, warp_row_offset, lane_id);

    // 5. 初始化 output 寄存器
    float acc_o[16][4] = {0.0f};

    // m_i 和 d_i 是维护更新 局部 exp 最大值 和 分母
    float m_i[2] = {-FLT_MAX, -FLT_MAX};
    float d_i[2] = {0.0f, 0.0f};
    float row_scale[2] = {1.0f, 1.0f}; // 用于 lazy rescale

    const int k_end = min(kv_len, q_start_idx + BM); // causal mask，只计算对角线和左边的 block
    int phase_k = 0;
    int phase_v = 0;
    const float scale_log2 = scale * 1.44269504f; // scale*log2(e)
    // 当小于给定阈值 2^-8 时，就对 acc_o 进行缩放。
    constexpr float lazy_scale_threshold = 0x1p-8f;

    // 6. kv loop
    for (int n = 0; n < k_end; n += BN) {
        // --- 6.1 TMA 异步加载 KV ---
        if (tid == 0) {
            mbarrier_expect_tx(mbar_k, BN * HEAD_DIM * sizeof(T));
            mbarrier_expect_tx(mbar_v, BN * HEAD_DIM * sizeof(T));
            cp_async_bulk_tensor_4d(mbar_k, &tma_k, Ks[0], 0, kv_head_id, n, batch_id);
            cp_async_bulk_tensor_4d(mbar_k, &tma_k, Ks[1], 64, kv_head_id, n, batch_id);
            cp_async_bulk_tensor_4d(mbar_v, &tma_v, Vs[0], 0, kv_head_id, n, batch_id);
            cp_async_bulk_tensor_4d(mbar_v, &tma_v, Vs[1], 64, kv_head_id, n, batch_id);
        }
        __syncthreads();//等 tid0 指令发射完毕
        mbarrier_wait(mbar_k, phase_k); // 这里只等 k 加载完毕即可，v 的加载和 qk 的计算并行
        phase_k ^= 1; // 反转 phase

        // --- 6.2 计算 S = Q * K^T ---
        float acc_s[8][4] = {0.f};

#pragma unroll
        for (int k_step = 0; k_step < 8; ++k_step) {
            uint32_t reg_k[8][2];
            ldmatrix_Ks<BN, HEAD_DIM>(reg_k, Ks, lane_id, k_step);
            mma_compute<8>(acc_s, reg_q[k_step], reg_k);
        }

        // --- 6.3 online softmax （更新 m_i、d_i，缩放校准） ---
        int row_0 = warp_row_offset + (lane_id / 4);
        int row_1 = row_0 + 8;

        float m_prev[2] = {m_i[0], m_i[1]};
        float m_curr[2] = {-FLT_MAX, -FLT_MAX};
        float local_d[2] = {0.0f, 0.0f};
        float2 p_row0[8];
        float2 p_row1[8];

        bool need_causal_mask = (n + BN > q_start_idx);
#pragma unroll
        for (int n_step = 0; n_step < 8; ++n_step) {
            int col_base = n_step * 8 + (lane_id % 4) * 2;

            if (need_causal_mask) {
                acc_s[n_step][0] = (q_start_idx + row_0 >= n + col_base) ? acc_s[n_step][0] : -FLT_MAX;
                acc_s[n_step][1] = (q_start_idx + row_0 >= n + col_base + 1) ? acc_s[n_step][1] : -FLT_MAX;
                acc_s[n_step][2] = (q_start_idx + row_1 >= n + col_base) ? acc_s[n_step][2] : -FLT_MAX;
                acc_s[n_step][3] = (q_start_idx + row_1 >= n + col_base + 1) ? acc_s[n_step][3] : -FLT_MAX;
            }

            m_curr[0] = fmaxf(m_curr[0], fmaxf(acc_s[n_step][0], acc_s[n_step][1]));
            m_curr[1] = fmaxf(m_curr[1], fmaxf(acc_s[n_step][2], acc_s[n_step][3]));
        }

#pragma unroll
        for (int i = 1; i < 4; i *= 2) {
            m_curr[0] = fmaxf(m_curr[0], __shfl_xor_sync(0xffffffff, m_curr[0], i));
            m_curr[1] = fmaxf(m_curr[1], __shfl_xor_sync(0xffffffff, m_curr[1], i));
        }
        m_curr[0] = (m_curr[0] == -FLT_MAX) ? -FLT_MAX : m_curr[0] * scale_log2;
        m_curr[1] = (m_curr[1] == -FLT_MAX) ? -FLT_MAX : m_curr[1] * scale_log2;

        m_i[0] = fmaxf(m_prev[0], m_curr[0]);
        m_i[1] = fmaxf(m_prev[1], m_curr[1]);

        float exp_mprev_mnew[2] = {exp2f(m_prev[0] - m_i[0]), exp2f(m_prev[1] - m_i[1])};
        float alpha_0 = (m_prev[0] == -FLT_MAX) ? 1.0f : exp_mprev_mnew[0];
        float alpha_1 = (m_prev[1] == -FLT_MAX) ? 1.0f : exp_mprev_mnew[1];

        row_scale[0] *= alpha_0;
        row_scale[1] *= alpha_1;

        if (row_scale[0] < lazy_scale_threshold) {
            lazy_rescale(acc_o, d_i, row_scale, 0);
        }
        if (row_scale[1] < lazy_scale_threshold) {
            lazy_rescale(acc_o, d_i, row_scale, 1);
        }

        float inv_row_scale_0 = __frcp_rn(row_scale[0]);
        float inv_row_scale_1 = __frcp_rn(row_scale[1]);

        // 6.4 计算 Ps
#pragma unroll
        for (int n_step = 0; n_step < 8; ++n_step) {
            float e_0_0 = exp2f(fmaf(acc_s[n_step][0], scale_log2, -m_i[0]));
            float e_0_1 = exp2f(fmaf(acc_s[n_step][1], scale_log2, -m_i[0]));
            float e_1_0 = exp2f(fmaf(acc_s[n_step][2], scale_log2, -m_i[1]));
            float e_1_1 = exp2f(fmaf(acc_s[n_step][3], scale_log2, -m_i[1]));

            p_row0[n_step] = {e_0_0 * inv_row_scale_0, e_0_1 * inv_row_scale_0};
            p_row1[n_step] = {e_1_0 * inv_row_scale_1, e_1_1 * inv_row_scale_1};

            local_d[0] += p_row0[n_step].x + p_row0[n_step].y;
            local_d[1] += p_row1[n_step].x + p_row1[n_step].y;
        }

#pragma unroll
        for (int i = 1; i < 4; i *= 2) {
            local_d[0] += __shfl_xor_sync(0xffffffff, local_d[0], i);
            local_d[1] += __shfl_xor_sync(0xffffffff, local_d[1], i);
        }

        d_i[0] += local_d[0];
        d_i[1] += local_d[1];

        // --- 6.5 O = P * V ---
        mbarrier_wait(mbar_v, phase_v);
        phase_v ^= 1;
#pragma unroll
        for (int k_step = 0; k_step < 4; ++k_step) {
            uint32_t reg_p[4]; // 构造 Fragment A
            reg_p[0] = pack_bfloat2(p_row0[k_step * 2 + 0]);
            reg_p[1] = pack_bfloat2(p_row1[k_step * 2 + 0]);
            reg_p[2] = pack_bfloat2(p_row0[k_step * 2 + 1]);
            reg_p[3] = pack_bfloat2(p_row1[k_step * 2 + 1]);

            uint32_t reg_v[16][2];
            ldmatrix_Vs<BN, HEAD_DIM>(reg_v, Vs, lane_id, k_step);
            mma_compute<16>(acc_o, reg_p, reg_v);
        }

        __syncthreads();
    }

    epilogue_writeback<BM, HEAD_DIM, THREADS_PER_BLOCK>(
        acc_o, m_i, d_i, Os, o, warp_row_offset, lane_id, q_start_idx, q_len, q_head, batch_id, head_id);
}
```

其中一些 ldmatrix，mma 等我都封装成函数了，方便理解整体流程，当然 online softmax 写的依然不是很好。其实除了上面流程里提到的。在整个实现里，还是有一些值得提一下的

### lazy rescale

理论上 online softmax 在每次循环都要对 acc_o 进行局部最大值的修正，但是我用了两个单独的寄存器专门存放 scale，等到达到某一定阈值时，再把 row_scale 乘进 acc_o 里。这里减轻了很多 cuda core 的压力，避免 tensor-core 计算流被打断。

这里阈值选了 2^-8，为什么这么选呢？因为 Ps 矩阵最后要转成 BF16 喂给 Tensor Core。BF16 的尾数（Mantissa）只有 7 位。一旦 row_scale 低于 2^-8，意味着倒数会放大 256 倍，此时 BF16 的最小精度刻度（ULP）会变成 2，彻底丢失小数部分（当然，在 softmax 作用之下会实际数值会相对小一些）。卡在 2^-8，是在极热循环分支开销和 BF16 精度之间，权衡出来的一个值。

### k_end 和 need_causal_mask

在 prefill 阶段，我们都知道 attention mask 是一个下三角矩阵，对角线右边其实概率都为 0，可以完全不参与计算，对角线左边都是完整参与 online softmax 过程的，只有对角线上的块才需要进行复杂的判断（是否要直接抹为最小值）。

### warp reduce m_i/d_i

从代码从可以看到，我们仅仅使用了两步的 warp reduce(__shfl_xor_sync) 就完成了局部最大值和分母和的 reduce 计算。因为在 m16n8k16 的 Fragment C 布局中，一个 warp 的 32 个线程被分为 8 组（每组 4 人）。这 4 个线程刚好瓜分了某两行的 8 个列元素。所以我们只需要用两步蝶形变换__shfl_xor_sync（掩码为 1 和 2），就能在一个 4 线程的小组内完成一整行的局部归约。

### Ps 寄存器重排

其实第一版我是把 Ps 写回 smem（利用 Ks 做中转 buffer），再用 ldmatrix.x4 读取出来。但是在我验证了 mma 16x8x16 的 Fragment C 和 Fragment A 的寄存器状态后，我发现，不知道是巧合还是 NV 的芯片设计如此：

我根本不需要写回 smem 再用 ldmatrix 读取。每个线程 hold 的 mma Fragment C 已经包含了下一个矩阵乘法的 Fragment A 所需的所有值。（NV 很可能早就为三个矩阵连乘的场景埋好了底层架构的伏笔）。

我们来推演一下：

- `Qs@Ks mma` 之后：
  - 在单一 `mma 16x8x16` 的视角下，Fragment C 的线程 0 hold 着：c[0][0], c[0][1], c[8][0], c[8][1]。
  - 但在整个 64x64x128 Tile 的外层循环视角下，线程 0 还会随着步长依次 hold 住：c[0][8], c[0][9], c[8][8], c[8][9]；以及 c[0][16], c[0][17], c[8][16], c[8][17]... 共计 8 组数据。
- 第二个 `Ps@Vs mma`：Fragment A 要求输入时，线程 0 必须 hold 住：a[0][0], a[0][1], a[8][0], a[8][1] 和 a[0][8], a[0][9], a[8][8], a[8][9]。

![]()

对比就能发现，它们的逻辑坐标是完全同构的！因此，我们可以直接从 Fragment C 的 acc_s 寄存器中取出 `Ps@Vs` 的 Fragment A 数值。当然要压缩一下数据类型，使用 `__float22bfloat162_rn` 将逻辑相邻的两个 float 原位压缩（Pack）成一个 bf162 向量即可。

这里 Fragment 具体排布之前也提过，就不重复提了，具体还是见 nv 的文档：<https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-Fragment-mma-16816-float>

### Vs 和 qk 计算的并行

最开始，我是昏了头了，Ks 和 Vs 一起加载到 smem。但是我忽然发现，qk 的计算和 softmax 过程和 Vs 毫无关系，这两部分是可以并行的，因此增加一个 mbarrier 用于 Vs 的同步，让 Vs 的加载和 qk 的计算并行（这也是我最终性能反超 FA2 的关键）

## 3. benchmark

不多说，直接上 benchmark 结果：

```yaml
####################################################################################################
prefill, batch:  1, seq: 512, head: 32, dim: 128
torch                                    mean time: 0.121095 ms, 17.73 tflops
fmha_tma_128                             mean time: 0.098975 ms, speedup: 1.22, tflops: 21.70
####################################################################################################
prefill, batch:  1, seq: 1024, head: 32, dim: 128
torch                                    mean time: 0.399812 ms, 21.48 tflops
fmha_tma_128                             mean time: 0.352873 ms, speedup: 1.13, tflops: 24.34
####################################################################################################
prefill, batch:  1, seq: 2048, head: 32, dim: 128
torch                                    mean time: 1.355930 ms, 25.34 tflops
fmha_tma_128                             mean time: 1.293091 ms, speedup: 1.05, tflops: 26.57
####################################################################################################
prefill, batch:  1, seq: 4096, head: 32, dim: 128
torch                                    mean time: 4.983902 ms, 27.58 tflops
fmha_tma_128                             mean time: 4.891181 ms, speedup: 1.02, tflops: 28.10
####################################################################################################
prefill, batch:  1, seq: 8192, head: 32, dim: 128
torch                                    mean time: 18.248972 ms, 30.13 tflops
fmha_tma_128                             mean time: 17.612578 ms, speedup: 1.04, tflops: 31.21
```

我们在所有 seq_len case 下都超越了 FA2 的性能，当然这主要归功于 TMA 解放了大量寄存器（主要是寻址变量），使得指令并行度更高，copy 和计算重叠也更充分

NCU report:
![summary](../static/fmha_summary.png)

我的算子寄存器还有充足的余量（FA2 已经拉满了）

![shared memory](../static/fmha_shared_table.png)

我的算子 L2 利用率拉爆了（90%+）

### 3.1 一些讨论

- 我的算子还有一些 non-fused 的 fp32 操作，基本都是 softmax 相关操作贡献的，我几度尝试减少单独的 add，mul 操作（比如用 exp 替换为 exp2(fmaf) 融合指令，lazy rescale），
  - 但终究无法完全干掉单独的乘法和加法，除非强行 fmaf(a, b, 0) 或者 fmaf(a,1,c), 但那毫无意义
- 同样的我的算子还有一点点 bank conflict，这里不是 swizzle 的静态地址映射有冲突，如果是地址映射冲突那样会很大的冲突比例，但我这里最大的 shared store 也只有 3%+
  - 我也不知道应该怎么解释这种现象，个人感觉是编译器把我的代码 unroll 开之后，“动态”的指令执行顺序所导致的
  - 为什么会有这个感觉，因为在我加入 lazy rescale 后，冲突量下降了 2/3（理论上 rescale 和 smem 完全无关，作为 smem 相关代码的 surrounding code，它影响到的只有指令执行顺序/间隔问题。加入 lazy rescale 时我还在 Ps 写回 Smem+ldmatrix 读取的写法，也可能和这块有关）
  - 同样的我把同一个循环内的两行交替访问 smem，换成两个单独行访问的循环，冲突量也降低了
  - 总之，目前的冲突量已经是我一些 trick 操作优化后的结果了
- FA 就没有 non-fused fp32 op，而且 0 bank 冲突（这就是为什么我说 flash attention 的 NCU profile 结果很漂亮）
  - 前者我不知道是如何做到的，不过我也不纠结，FA 调用了 cutlass 库实现，可能有对应到 sass 级别的优化，这是我做不到的。
  - 后者我倒是能理解，因为从 NCU 看，FA 采取了和我不同的 tiling 策略，其 grid 总量比我少一半，说明其 BM 很有可能是使用的 128，只有一个 block 活跃。
    - 在一个 block 内部通过 cp.async + ldmatrix 操作，这个衔接很丝滑，其 epilogue 写回肯定比我手写实现的更好，也同样能理解。
    - 所以 0 冲突很好理解
- 目前，由于我的一些 trick 操作，导致算子流程已经不是很直观了，还请看客担待。有空我会考虑重构下。

## TODO

attention 可做的事情就太多了，水很深。

比如：

- causal mask 逻辑优化：当前我用 need_causal_mask 做拦截，但 unroll 展开的指令肯定膨胀了，造成指令级并行度下降
- decode attention
  - prefill causal attention 是算力密集（拼 Tensor Core 和访存隐藏），而 decode 彻底变成了带宽瓶颈（GEMV 纯拼访存带宽）。从 mma 切换到如何极致压榨 L2 和 HBM 带宽，这是完全不同的优化哲学。
- attention with kv cache block
  - 这是 paged attention 核心，要想直接接入现有 LLM 推理框架，这种实现必不可少
- fp8 混合精度与 KV 量化
  - 拥抱 sm120 的 fp8 特性，并探索 INT8/INT4 KV Cache 量化，进一步优化访存

接下来我可能会先朝 decode attention 动手，但也可能会先放空休息一段时间。毕竟纯手搓拿下 flash attention 后，我觉得自己在 sm120 架构底层这块的探索，终于算是摸到了一个成熟的阶段。

## 结束

这就是我这个 sm120 fmha 的算子实现全部过程啦。

完整代码和测试脚本，还请从 github 获取：

以上
