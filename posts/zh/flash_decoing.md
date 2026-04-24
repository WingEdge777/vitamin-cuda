# [CUDA 优化实战] 纯手搓 flash decoding sm120 : 拉爆显存带宽的 cuda c++实现

## 0. 序 - flash decode 和 prefill attention : 完全不同的优化哲学

> 本文适用于有一定 CUDA 编程基础，熟悉 GEMM/multi-head-attention 优化，对进阶嵌入 PTX 指令性能调优感兴趣的读者阅读
>
> 完整 kernel 和测试代码可以点击 [flash_decode](/kernels/flash_decode) 查看，欢迎大家关注我的 vitamin-cuda 项目：

章接上文，上篇 fmha 文章中我们实现了 flash attention (fmha sm120)，并实现了超越 FA2 的性能，当然这主要归功于 TMA 的外挂加持。本文延续之前的内容，接下来给出 flash decoding 的实现。

对比的 baseline 我原本是想选择 flashinfer 的，但是 flashinfer 在我的 wsl 环境下跑不起来（报了一个除 0 异常），我也不知道怎么解决。最终 baseline 只好选了 pytorch.compile（依赖 triton） 加持的 native 实现。

```python
@torch.compile
def torch_native_decode(q, k, v, scale=None):
    # q: [head, dim] -> [32, 128]
    # k: [seq, head, dim] -> [4096, 32, 128]
    # v: [seq, head, dim] -> [4096, 32, 128]
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    # 调整维度以适应 Batched GEMV
    q_b = q.unsqueeze(1)  # [32, 1, 128]
    k_b = k.permute(1, 2, 0)  # [32, 128, 4096]
    v_b = v.transpose(0, 1)  # [32, 4096, 128]

    # S = Q @ K^T
    attn_scores = torch.matmul(q_b, k_b) * scale  # [32, 1, 4096]
    attn_probs = torch.softmax(attn_scores, dim=-1)

    # O = P @ V
    out = torch.matmul(attn_probs, v_b)  # [32, 1, 128]

    return out.squeeze(1)  # [32, 128]
```

但是大家不要看到说 pytorch 实现，就觉得性能很差，太低估 pytorch 了。我们之前就说过 decode attention 实质上退化为了 gemv（矩阵向量乘法），pytorch 的启发式调用 cuBLAS 的 gemv，叠加 torch.compile 优化，做一个 baseline 绰绰有余。你的 kernel 不认真写，还真不一定打得过。

本文的 kernel 大纲如下：

- flash_decode_tma_128 （BN=64，TMA + ldmatrix + mma）

依然只有一个算子（当然类似于 large scale softmax，用了 split-k 思想，实际上 launch 了两个 kernel)，支持 head_dim 为 128 情况下 mha 的 decode attention。

## 1. flash decoding

flash decoding 的思想大家肯定都学习过了。在 llm decoding 的阶段，由于 batchsize/q_seq_len 太小（甚至直接等于 1），attention 中对 q 的序列并行完全没了，无法充分利用所有 SM。因此考虑对 kv 的 seq 维度进行 chunk 切分，然后分两步完成 attention

- 一个 block 负责一个 q 和 kv chunk 的 attention，计算完 chunk 内的 m_i/d_i/acc_o，写回 gmem
- 读取上一步的中间结果，merge m_i/d_i/acc_o，得到最终的 o 输出

这张官方博客的示意图，相信关注的人看过没有十遍也有八遍了。但我们就不重复说原理性的东西，直直白白地讲清楚如何使用 c++ 代码纯手搓出来一个 flash decoding kernel.

![](https://pytorch.org/wp-content/uploads/2023/10/image.gif)

为了方便理解，我们这里仅考虑 q shape 为 [head, dim]， kv shape 为 [seq, head, dim] 的 一次 decode 计算（和上面的 pytorch 代码对应）

先说 data tiling 策略：

- 搬运 kv 数据 tile，这里沿用了 flash attention 实现中的 BN=64，即 64x128, 为什么？
- 我的卡一共 100KB smem，一个 block 最多 48KB，这里即使省掉了原来 q 中 BM 对应的 16KB，也不够 Ks 和 Vs 分的，暂时也不想搞个奇奇怪怪的 BN 大小

也就是说 kv chunk loop 中每次循环加载 64x128 的 Ks/Vs tile。

再说 thread block/grid 配置：

- block 直接定了 128，不要问为什么，多番测试（32，64 太小，block 数量有限，可切换的 warp 量太少，Occupancy 不足；256 太大，计算密度很低根本不需要那么多线程参与）后，这是最好的选择。
- grid 上，显然 q 失去了 seq 维度，无法并行。head 还是放在 y 维度上，再考虑对 kv 的 seq 进行切分放到 x 维度。这里只有一个切块大小的问题：
  - 我的 5060 只有 26 个 SM，为了充分利用 SM，我们保障 block 数量为 SM 数量的整数倍，不用太多，2~4 倍即可，我这里就用了 26x2，因此先确定预期的总 chunk 数为 52 个左右
  - 然后运行时用 head*seq/52，且向上对 2 的幂取整得到 chunk_size，则 grid.x = (seq + chunk_size - 1) / chunk_size;

ok，把以上思路整理一下，基本就确定了我们的 kernel launch 代码。

此外，我们这里考虑放弃 tensor core 的使用，为啥？因为 mma m16n8k16，要求 m16 行，我们 q 其实就是一行，强行 padding 到 16 行，除了浪费就是浪费。而且正如前边所说，decode attention 实质化退化为了 gemv，是访存瓶颈。用不用 tensor-core 都无所谓，需要的是快速大批量的发射访存请求以打满带宽。

决定不使用 mma 后，那么 ldmatrix 还有 TMA 的 swizzle 都可以省了。TMA 直接原样拷贝一整块 Ks/Vs tile（64x128）到 smem 即可，后面会说如何读取 smem

```cpp
inline int get_chunk_size(int q_head, int kv_len, int num_sms) {
    int target_blocks = num_sms * 2;

    // Total_Blocks = q_head * (kv_len / chunk_size)
    // chunk_size = (q_head * kv_len) / target_blocks
    int chunk = (q_head * kv_len) / target_blocks;

    if (chunk <= 256)
        return 256;
    if (chunk <= 512)
        return 512;
    if (chunk <= 1024)
        return 1024;
    return 2048;
}

#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

template <typename T>
inline CUtensorMap create_3d_tensor_map(T *global_address,
                                        uint64_t dim_d,
                                        uint64_t dim_h,
                                        uint64_t dim_s,
                                        uint64_t stride_h,
                                        uint64_t stride_s,
                                        uint32_t box_d,
                                        uint32_t box_s) // Each kernel load takes a (box_s x box_d) block
{
    CUtensorMap tmap;
    CUtensorMapDataType type =
        std::is_same_v<T, __half> ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;

    // TMA dimensions: from fastest (0) to slowest (2)
    uint64_t globalDim[3] = {dim_d, dim_h, dim_s};

    // globalStrides are strides for dimensions 1, 2, must be in Bytes
    uint64_t globalStrides[2] = {stride_h, stride_s};

    uint32_t boxDim[3] = {box_d, 1, box_s};
    uint32_t elementStrides[3] = {1, 1, 1};

    CUresult res = cuTensorMapEncodeTiled(&tmap,
                                          type,
                                          3, // Rank = 3
                                          global_address,
                                          globalDim,
                                          globalStrides,
                                          boxDim,
                                          elementStrides,
                                          CU_TENSOR_MAP_INTERLEAVE_NONE,
                                          swizzle,
                                          CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    TORCH_CHECK(res == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed for 3D Tensor!");
    return tmap;
}

#define DISPATCH_TMA_KERNEL(NAME, HEAD_DIM, CHUNK_SIZE)                                                                \
    NAME##_kernel<BN, CHUNK_SIZE, HEAD_DIM, 128, __nv_bfloat16>                                                        \
        <<<blocks_per_grid, 128, smem_bytes, stream>>>(reinterpret_cast<__nv_bfloat16 *>(q.data_ptr()),                \
                                                       tma_k,                                                          \
                                                       tma_v,                                                          \
                                                       reinterpret_cast<float *>(ws_o.data_ptr()),                     \
                                                       reinterpret_cast<float *>(ws_lse.data_ptr()),                   \
                                                       kv_len,                                                         \
                                                       q_head,                                                         \
                                                       kv_head,                                                        \
                                                       scale);

#define binding_tiled_tma_func_gen(name, HEAD_DIM)                                                                     \
    void name##_##HEAD_DIM(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, float scale) {          \
                                                                                                                       \
        CHECK_T(q);                                                                                                    \
        CHECK_T(k);                                                                                                    \
        CHECK_T(v);                                                                                                    \
        CHECK_T(o);                                                                                                    \
                                                                                                                       \
        /* Extract dimension info dynamically from Tensor */                                                           \
        const int q_head = q.size(0);                                                                                  \
        const int head_dim = q.size(1);                                                                                \
        const int kv_len = k.size(0);                                                                                  \
        const int kv_head = k.size(1);                                                                                 \
                                                                                                                       \
        /* Only validate that head_dim matches the compile-time constant */                                            \
        TORCH_CHECK(head_dim == HEAD_DIM, "Head dim mismatch: expected ", HEAD_DIM);                                   \
                                                                                                                       \
        int elem_bytes = k.element_size();                                                                             \
        uint64_t k_stride_h = k.stride(1) * elem_bytes;                                                                \
        uint64_t k_stride_s = k.stride(0) * elem_bytes;                                                                \
        uint64_t v_stride_h = v.stride(1) * elem_bytes;                                                                \
        uint64_t v_stride_s = v.stride(0) * elem_bytes;                                                                \
                                                                                                                       \
        const int BN = 64;                                                                                             \
        const int num_sms = 26;                                                                                        \
        const size_t smem_bytes = BN * head_dim * sizeof(__nv_bfloat16) * 2 + sizeof(mbarrier_t) * 2;                  \
        const int chunk_size = get_chunk_size(q_head, kv_len, num_sms);                                                \
        CUtensorMap tma_k = create_3d_tensor_map<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16 *>(k.data_ptr()),       \
                                                                head_dim,                                              \
                                                                kv_head,                                               \
                                                                kv_len,                                                \
                                                                k_stride_h,                                            \
                                                                k_stride_s,                                            \
                                                                head_dim,                                              \
                                                                BN);                                                   \
                                                                                                                       \
        CUtensorMap tma_v = create_3d_tensor_map<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16 *>(v.data_ptr()),       \
                                                                head_dim,                                              \
                                                                kv_head,                                               \
                                                                kv_len,                                                \
                                                                v_stride_h,                                            \
                                                                v_stride_s,                                            \
                                                                head_dim,                                              \
                                                                BN);                                                   \
                                                                                                                       \
        TORCH_CHECK(q_head % kv_head == 0, "q_head must be divisible by kv_head");                                     \
        const dim3 blocks_per_grid((kv_len + chunk_size - 1) / chunk_size, q_head);                                    \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());                               \
        auto ws_lse = torch::empty({q_head, blocks_per_grid.x}, options);                                              \
        auto ws_o = torch::empty({q_head, blocks_per_grid.x, head_dim}, options);                                      \
        /* launch kernel */                                                                                            \
        switch (chunk_size) {                                                                                          \
            case 256: DISPATCH_TMA_KERNEL(name, HEAD_DIM, 256); break;                                                 \
            case 512: DISPATCH_TMA_KERNEL(name, HEAD_DIM, 512); break;                                                 \
            case 1024: DISPATCH_TMA_KERNEL(name, HEAD_DIM, 1024); break;                                               \
            case 2048: DISPATCH_TMA_KERNEL(name, HEAD_DIM, 2048); break;                                               \
            default: TORCH_CHECK(false, "Unsupported chunk size: ", chunk_size);                                       \
        }                                                                                                              \
        flash_decode_reduce_kernel<HEAD_DIM, 128, __nv_bfloat16>                                                       \
            <<<q_head, 128, 0, stream>>>(reinterpret_cast<float *>(ws_o.data_ptr()),                                   \
                                         reinterpret_cast<float *>(ws_lse.data_ptr()),                                 \
                                         reinterpret_cast<__nv_bfloat16 *>(o.data_ptr()),                              \
                                         blocks_per_grid.x);                                                           \
    }

binding_tiled_tma_func_gen(flash_decode_tma, 128);

#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // flash_decode_tma_128
    torch_pybinding_func(flash_decode_tma_128);
}
```

使用一个 help func 拿到 chunk_size，根据 chunk size 用一个 dispatch 宏分发到不同的 kernel launch 上（因为我们需要编译期确定 chunk_size，以帮助 kernel 内部循环展开）

## 2. kernel 实现细节

从逻辑上来说，一次 tiling 下的数据（chunk 大小是 tiling 大小的整数倍），我们需要加载一行 Qs[1,128]，然后和 Ks.T[128,64] 做向量矩阵乘法得到 s[1,64], 然后 softmax 完得到 Ps[1,64], 再乘以 Vs[64，128]。

现在的情况是，我们一个 block 有 4 个 warp，128 线程，如何来瓜分这些计算。

首先排除掉一个线程负责一整行/一整列点积的思路，这不是 cuda 并行编程的第一性原理，不能用串行思维去写代码，这样直接后果就是 bank conflict 爆炸，而且算完一行的点积结果后，还是要经过 block 线程间进行同步广播，否则无法参数接下来 Vs 的计算。

我们应该先从一个 warp 去考虑，比如一个 warp 负责一行计算，那么 128 的向量共 32 线程，每个线程就只需要负责 4 个元素。4 个 warp，每个 warp 负责 16 行，这样 16 行 16 次循环内只需要 warp 间同步，等 16 行计算完四组再进行 block 间同步。

更进一步，考虑到我们使用的是半精度，也就是说一个元素才 2 字节，我们为了最大化压榨带宽，肯定是用 float4 向量化指令，因此我们让一个线程负责 8 个元素，一个 warp 一次就可以算两行，只需要循环 8 次。

初步整理一下算法流程如下：

- kernel pass 1：
  - 初始化 Ks、Vs 2 块 smem 和 2 个 tma mbarrier
  - 初始化 acc_o[8], 每个 group（16 线程）私有化初始化历史状态 acc_o[8], m_i, d_i
  - 加载 Qs[8](使用 float4 向量化加载到寄存器)
  - 一个 block 负责一个 chunk，kv chunk 内 loop：
    - tma 发起加载 Ks、Vs，并等待 Ks 加载完成
    - [计算 S] 循环 8 次（Group 内每线程负责 8 行）：
      - float4 向量化读取 Ks 一行内的 8 个元素
      - Qs[8] 和 reg_k[8] 进行点积计算
      - Warp Reduce 求和得到单行的 Attention Score，并统计当前小块的 m_part
    - wait Vs 加载完毕
    - [计算 O] 循环 8 次：
      - 根据 m_part 计算当前行的 Softmax 权重标量 p
      - p 乘以 reg_v[8] 向量，累加得到当前小块的 part_o[8]，并统计当前小块的 d_part
    - [group 内部状态更新] 使用安全的 online softmax 逻辑计算 alpha，将当前的 part_o、d_part、m_part 融合进 group 维护的历史 acc_o、d_i、m_i 中
  - block 内各 group 进行 block_reduce，合并各自的 acc_o、d_i、m_i
  - 将最终合并的寄存器结果转化为 ws_o 和 d_i/m_i，写回 gmem
- kernel pass 2
  - 读取上面 gmem 的 ws_o，和 m_i/d_i，online softmax 继续规约得到最终输出 o
- 结束

上面是最原汁原味的分块 online + “offline” softmax attention. 但参考一些教程，可以加入一个 trick 优化，将 m_i 和 d_i 合并为 lse（logsumexp），就是对 e 的指数和再取对数。当然由于我们还加入了 log2_scale 的技巧，因此 lse 变成了一个有点丑陋的东西：

```c++
lse(o) = m_i * ln(2) + ln(d_i)
```

然后 pass 2 内通过 lse 和 ws_o 合并方式伪代码为：

```c++
float max_lse = max(lse_i);
float global_lse = max_lse + ln(sum(exp(lse_i - max_lse)));
o = sum(ws_o_i * exp(lse_i - global_lse));
```

上 kernel 代码：

```cpp
// flash decoding softmax(q @ k.T*scale) @ v
template <const int BN = 64,
          const int CHUNK_SIZE = 256,
          const int HEAD_DIM = 128,
          const int THREADS_PER_BLOCK = 128,
          typename T>
__global__ void flash_decode_tma_kernel(T *q,
                                        const __grid_constant__ CUtensorMap tma_k,
                                        const __grid_constant__ CUtensorMap tma_v,
                                        float *ws_o,   // [q_head, num_chunks, HEAD_DIM]
                                        float *ws_lse, // [q_head, num_chunks]
                                        int kv_len,
                                        int q_head,
                                        int kv_head,
                                        float scale) {
    static_assert(THREADS_PER_BLOCK == 128);
    static_assert(BN == 64);

    // 1. shared memory: K tile, V tile, mbarriers
    extern __shared__ __align__(128) uint8_t smem_buf[];
    T(*Ks)[HEAD_DIM] = reinterpret_cast<T(*)[HEAD_DIM]>(smem_buf);
    T(*Vs)[HEAD_DIM] = reinterpret_cast<T(*)[HEAD_DIM]>(smem_buf + BN * HEAD_DIM * sizeof(T));
    mbarrier_t *mbar_k = reinterpret_cast<mbarrier_t *>(smem_buf + BN * HEAD_DIM * sizeof(T) * 2);
    mbarrier_t *mbar_v = mbar_k + 1;

    // 2. coordinates
    const int tid = threadIdx.x;
    const int chunk_id = blockIdx.x;
    const int q_head_id = blockIdx.y;
    const int kv_group_size = q_head / kv_head;
    const int kv_head_id = q_head_id / kv_group_size;

    constexpr int THREADS_PER_ROW = 16;
    constexpr int NUM_GROUPS = THREADS_PER_BLOCK / THREADS_PER_ROW;
    constexpr int ROWS_PER_GROUP = BN / NUM_GROUPS;
    const int group_id = tid / THREADS_PER_ROW;
    const int lane_id = tid % THREADS_PER_ROW;

    if (tid == 0) {
        mbarrier_init(mbar_k, 1);
        mbarrier_init(mbar_v, 1);
    }
    __syncthreads();

    // 3. load q fragment
    pack128 qs{FLOAT4(q[q_head_id * HEAD_DIM + lane_id * 8])};

    // 4. init subgroup-local online softmax state
    __align__(16) float acc_o[8] = {0.0f};
    float m_i = -FLT_MAX;
    float d_i = 0.0f;

    int phase_k = 0;
    int phase_v = 0;
    const float scale_log2 = scale * 1.44269504f; // scale*log2(e)
    const int num_chunks = gridDim.x;
    const int chunk_start = chunk_id * CHUNK_SIZE;
    const int chunk_end = min(chunk_start + CHUNK_SIZE, kv_len);

    // 5. loop over KV tiles inside this chunk
    for (int n = chunk_start; n < chunk_end; n += BN) {
        int current_bn = min(BN, chunk_end - n);

        // 5.1 TMA async load K/V
        if (tid == 0) {
            mbarrier_expect_tx(mbar_k, BN * HEAD_DIM * sizeof(T));
            mbarrier_expect_tx(mbar_v, BN * HEAD_DIM * sizeof(T));
            cp_async_bulk_tensor_3d(mbar_k, &tma_k, Ks, 0, kv_head_id, n);
            cp_async_bulk_tensor_3d(mbar_v, &tma_v, Vs, 0, kv_head_id, n);
        }
        __syncthreads();
        mbarrier_wait(mbar_k, phase_k);
        phase_k ^= 1; // flip phase

        // 5.2 compute S = Q * K^T, keep rows per subgroup in registers
        const int row_begin = group_id * ROWS_PER_GROUP;
        float acc_s[ROWS_PER_GROUP];
        float m_part = -FLT_MAX;
#pragma unroll
        for (int i = 0; i < ROWS_PER_GROUP; ++i) {
            acc_s[i] = -FLT_MAX;
        }
#pragma unroll
        for (int i = 0; i < ROWS_PER_GROUP; ++i) {
            const int row = row_begin + i;
            float sum = 0.0f;
            if (row < current_bn) {
                pack128 ks{FLOAT4(Ks[row][lane_id * 8])};
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    sum += static_cast<float>(qs.bf[j]) * static_cast<float>(ks.bf[j]);
                }
            }
            sum = warp_reduce_sum<THREADS_PER_ROW>(sum);
            if (row < current_bn) {
                acc_s[i] = sum * scale_log2;
                m_part = fmaxf(m_part, acc_s[i]);
            }
        }

        // 5.3 accumulate subgroup-local O = P * V
        mbarrier_wait(mbar_v, phase_v);
        phase_v ^= 1;
        float part_d = 0.0f;
        float part_o[8] = {0.0f};
#pragma unroll
        for (int i = 0; i < ROWS_PER_GROUP; ++i) {
            const int row = row_begin + i;
            if (row < current_bn) {
                float p = exp2f(acc_s[i] - m_part);
                part_d += p;

                pack128 vs{FLOAT4(Vs[row][lane_id * 8])};
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    part_o[j] += p * static_cast<float>(vs.bf[j]);
                }
            }
        }
        if (m_part != -FLT_MAX) {
            const float m_new = fmaxf(m_i, m_part);
            const float alpha_old = exp2f(m_i - m_new);
            const float alpha_new = exp2f(m_part - m_new);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc_o[i] = acc_o[i] * alpha_old + part_o[i] * alpha_new;
            }
            d_i = d_i * alpha_old + part_d * alpha_new;
            m_i = m_new;
        }
    }

    // 6. merge subgroup states once per chunk, then write split results
    const float m_chunk = block_reduce_max<NUM_GROUPS, THREADS_PER_ROW>(lane_id == 0 ? m_i : -FLT_MAX);
    const float alpha = d_i > 0.0f ? exp2f(m_i - m_chunk) : 0.0f;
    const float d_chunk = block_reduce_sum<NUM_GROUPS, THREADS_PER_ROW>(lane_id == 0 ? d_i * alpha : 0.0f);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        acc_o[i] = block_reduce_sum_by_lane<NUM_GROUPS, THREADS_PER_ROW>(acc_o[i] * alpha);
    }

    if (group_id == 0) {
        int out_base_idx = (q_head_id * num_chunks + chunk_id) * HEAD_DIM + lane_id * 8;
        float inv_d = __frcp_rn(d_chunk);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            acc_o[i] *= inv_d;
        }
        pack128 out_pack0, out_pack1;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            out_pack0.f[i] = acc_o[i];
            out_pack1.f[i] = acc_o[i + 4];
        }
        FLOAT4(ws_o[out_base_idx + 0]) = out_pack0.f4;
        FLOAT4(ws_o[out_base_idx + 4]) = out_pack1.f4;

        if (lane_id == 0) {
            int scalar_idx = q_head_id * num_chunks + chunk_id;
            ws_lse[scalar_idx] = m_chunk * 0.6931471805599453f + logf(d_chunk);
        }
    }
}

template <const int HEAD_DIM = 128, const int THREADS_PER_BLOCK = 128, typename T>
__global__ void flash_decode_reduce_kernel(float *ws_o, float *ws_lse, T *o, int num_chunks) {
    const int q_head_id = blockIdx.x;
    const int tid = threadIdx.x;
    constexpr int NUM_WARPS = THREADS_PER_BLOCK / WARP_SIZE;

    __shared__ float s_lse;

    float lse_max = -FLT_MAX;
    for (int chunk = tid; chunk < num_chunks; chunk += THREADS_PER_BLOCK) {
        lse_max = fmaxf(lse_max, ws_lse[q_head_id * num_chunks + chunk]);
    }
    lse_max = block_reduce_max<NUM_WARPS, WARP_SIZE>(lse_max);

    float lse_sum = 0.0f;
    for (int chunk = tid; chunk < num_chunks; chunk += THREADS_PER_BLOCK) {
        lse_sum += expf(ws_lse[q_head_id * num_chunks + chunk] - lse_max);
    }
    lse_sum = block_reduce_sum<NUM_WARPS, WARP_SIZE>(lse_sum);
    if (tid == 0) {
        s_lse = logf(lse_sum) + lse_max;
    }
    __syncthreads();

    const int col = tid * 8;
    if (col >= HEAD_DIM) {
        return;
    }

    float out[8] = {0.0f};
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const int scalar_idx = q_head_id * num_chunks + chunk;
        const float weight = expf(ws_lse[scalar_idx] - s_lse);
        const int base_idx = scalar_idx * HEAD_DIM + col;
        pack128 partial0{FLOAT4(ws_o[base_idx + 0])};
        pack128 partial1{FLOAT4(ws_o[base_idx + 4])};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            out[i] += partial0.f[i] * weight;
            out[i + 4] += partial1.f[i] * weight;
        }
    }

    pack128 out_pack;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        out_pack.bf[i] = __float2bfloat16_rn(out[i]);
    }
    FLOAT4(o[q_head_id * HEAD_DIM + col]) = out_pack.f4;
}
```

其中 warp reduce、block reduce、tma copy 等都抽成函数出去了。

细心的朋友可能注意到了

- flash decoding 里没有 lazy rescale，因为没必要，原来 prefill 里 是用 16 次 Ps scale 去替换 64 次 acc_o 的 scale 是值得的，这里 acc_o 只有 8，和 p 乘以 inv_scale 的次数相同。
- 同样的，也没有 need_casual_mask 校验，因为对于当前的 q，所以 kv 都是可见的

唯一值得说明一下的是

### grouped warp/block reduce

因为我们一个 warp 负责两行，相当于把一个 warp 劈开成两半，分别去做 reduce 了，所以 warp reduce 的时候要指定宽度 16，用了个 group_size 模板参数

其实本版代码的实现还是很粗糙的，比如对于 ws_o/ws_lse 的写回还没有做优化，后续再看吧~（stay tuned，我也可能去学习一下 flashinfer 里如何实现等等）

## 3. benchmark

不多说，直接上 benchmark 结果：

```yaml
####################################################################################################
prefill, kv seq: 8192, head: 32, dim: 128
torch.compile                            mean time: 0.458745 ms, 292.61 GB/s
flash_decode_tma_128                     mean time: 0.446513 ms, speedup: 1.03, GB/s: 300.63
####################################################################################################
prefill, kv seq: 16384, head: 32, dim: 128
torch.compile                            mean time: 0.878726 ms, 305.50 GB/s
flash_decode_tma_128                     mean time: 0.778402 ms, speedup: 1.13, GB/s: 344.88
####################################################################################################
prefill, kv seq: 10240, head: 32, dim: 128
torch.compile                            mean time: 0.611377 ms, 274.44 GB/s
flash_decode_tma_128                     mean time: 0.520998 ms, speedup: 1.17, GB/s: 322.05
####################################################################################################
prefill, kv seq: 65536, head: 32, dim: 128
torch.compile                            mean time: 2.937899 ms, 365.49 GB/s
flash_decode_tma_128                     mean time: 2.920613 ms, speedup: 1.01, GB/s: 367.65
####################################################################################################
prefill, kv seq: 131072, head: 32, dim: 128
torch.compile                            mean time: 5.873622 ms, 365.62 GB/s
flash_decode_tma_128                     mean time: 5.720886 ms, speedup: 1.03, GB/s: 375.38
####################################################################################################
prefill, kv seq: 131073, head: 32, dim: 128
torch.compile                            mean time: 5.959847 ms, 360.33 GB/s
flash_decode_tma_128                     mean time: 5.718791 ms, speedup: 1.04, GB/s: 375.52
```

可以看到，性能上超过了 torch.compile 的 native 实现，但强的有限（pytorch+compile 真的不弱）。

另外 flashinfer 我没有跑通，所以无从比较，flashinfer 可能会比我这个 kernel 强的。因为虽然大 seq 下我的算子逻辑带宽使用率（卡理论峰值带宽 384GB/s），已经达到了 375.52⁄384 = 97.8%（巨高了），但短序列情况下，离峰值还差一些的。

ncu report：

![](../static/flash_decoding_summary.png)
![](../static/flash_decoding_detail.png)

有一些 uncoalesced global accesses (ws_o 和 ws_lse 写回没做优化，但这已不在热点循环内，对对整体性能影响微乎其微。)，此外 DRAM 带宽使用率硬件统计也拉到 90%+了。

## 4. TODO

之前列的 TODO，已经完成了 decode attention，当然这个 kernel 我还在考虑进一步优化 （还有相对冗余的 smem；换单 block 二/三级流水线效果会更好）。

此外，剩下 attention with kv cache，感觉手搓有点恶心，就是为了对齐数据边界而写代码，还是算了。量化 attention 还是要结合模型和具体 case 进行考虑，也放弃了。

后续可能会开始向更平移近人的 triton 等 python dsl 或者 更叫人头晕的 cutlass/cute 前进？

## 5. 结束

以上就是我目前对 flash decoding 的所有理解啦（but stay tuned！）

如有错误，欢迎指正。如有建议，也欢迎讨论（真心求教，因为感觉写的还有问题）

完整代码和测试脚本，还请从 github 获取：

以上
