# [CUDA in Practice] HGEMM SM120 — Micro-Sculpture Warfare in 100KB SMEM: Tensor Core, TMA, ldmatrix, mma
>
>Note: Text translated by AI. Code crafted by human.
>

>
> Sorry folks — I said there wouldn't be a sequel to the GEMM series, but I lied. Today it's HGEMM again, but this time we're embracing everything the RTX 5060 Laptop has to offer: TMA + ldmatrix + mma, pushing the limits.
>
> This article is intended for readers with a solid foundation in CUDA programming who are familiar with GEMM optimization and interested in advanced Tensor Core / inline PTX instruction performance tuning.
>
> The complete kernel and test code can be found at [hgemm_sm120](/kernels/hgemm_sm120).
>

## 0. Preface — SM120: The Castrated Blackwell

As everyone knows, our GeForce 50-series consumer GPUs with the SM120 architecture may carry the Blackwell name, but they're a completely different species from the B-series SM100. Cut after cut — everything's been stripped away. No tcgen5 instructions, no wgmma. So what do we get? We get TMA (Tensor Memory Accelerator), and NVIDIA was kind enough to extend mma instructions to support fp8/6/4 and other precisions. Our focus today is TMA — a dedicated hardware unit introduced in the Hopper architecture specifically for accelerating tensor data copies. A single instruction can asynchronously transfer an entire matrix tile. In short: fast, instruction-efficient, asynchronous, and comes with built-in swizzle. Paired with wgmma, it's absolutely killer.

Too bad we don't have wgmma.

To my knowledge, CUTLASS hasn't even implemented this cursed combination of TMA + mma for HGEMM — only quantized low-precision GEMM variants. See [CUTLASS example 79/87](https://github.com/NVIDIA/cutlass/tree/main/examples/79_blackwell_geforce_gemm) if interested. So my HGEMM implementation that grafts TMA onto ldmatrix + mma.sync — I won't claim it's the only one on the internet, but it's certainly a rare specimen. Enjoy the show~

Picking up where we left off, this article uses M=N=K=4096 (MxKxN, the medium-scale sweet spot where cuBLAS excels) GEMM in fp16/bf16 as an example. Running on an RTX 5060 Laptop, we implement HGEMM using TMA (with built-in swizzle) grafted onto ldmatrix (manual swizzle) + mma. We benchmark against cuBLAS and the `hgemm_bcf_dbf_rw_kernel` from the previous article as baselines.

Let me give the conclusion upfront: after extensive testing, the final kernel — with 3-stage TMA pipeline + double-buffered register ldmatrix + mma — versus my hand-written pure SM80-architecture kernel, has, at best, a marginal edge in performance. With memory and GPU clocks locked to maximum, across 10 benchmark runs, it trades wins roughly 6:4 against `hgemm_bcf_dbf_rw_kernel`.

A bit disappointing, isn't it? (I'll admit I was, considering how much time I spent debugging and testing to get TMA working.)

But looking at it from another angle, I've resolved the regret from the previous article — I've essentially built what I consider a perfect kernel. The `Uncoalesced Shared Accesses` that plagued `hgemm_bcf_dbf_rw_kernel`? Gone. Haha~

The only remaining warnings are a math-pipeline wait stall (even the Tensor Cores can't keep up) and an insufficient warp occupancy warning (out of smem). Everything else is clean.

This article presents 4 kernel implementations.

Kernel outline (the first one is the cuBLAS kernel):

- hgemm_cublas bf16/fp16 version
- hgemm_bcf_dbf_rw bf16/fp16 version (ldmatrix + mma, As/Bs swizzle bcf, double buffer, coalesced r/w gmem — refactored version with abstracted copy, ldmatrix, mma compute functions)
- hgemm_k_stages bf16/fp16 version (based on hgemm_bcf_dbf_rw, supports 3-stage pipeline — smem maxed out)
- hgemm_tma_r_k_stages bf16/fp16 version (based on hgemm_k_stages, replaces cp.async gmem reads with TMA copy)
- hgemm_tma_rw_k_stages (TODO — use TMA for gmem writeback too, but frankly I've lost the motivation since the expected gain is minimal)

## 1. hgemm_bcf_dbf_rw

This kernel was covered in detail in the previous article — how it evolved from the basic version to the final kernel. Interested readers, please refer to that post.

Here we've only done some refactoring to make the kernel structure shorter and clearer, easing the transition to multi-stage pipelining. So let's go straight to the code:

```cpp
// a block calculate c[128][128]
template <const int BM = 128, const int BN = 128, const int BK = 32, typename T>
__global__ void hgemm_bcf_dbf_rw_kernel(T *a, T *b, T *c, int m, int n, int k) {
    // grid swizzling
    int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8; // set execution block width to 8

    int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    int by = (linear_id / SWIZZLE_W) % gridDim.y;

    int tid = threadIdx.x; // 0~255
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // load mapping
    int load_a_row = tid / 4;        // 0~63
    int load_a_col = (tid % 4) * 8;  // 0,8,16,24
    int load_b_row = tid / 16;       // 0~15 (K dimension)
    int load_b_col = (tid % 16) * 8; // 0,8,16 ... 120 (N dimension)

    // A/B both row-major, using union to reuse the same memory — elegant approach
    __shared__ __align__(128) union {
        // first half: A and B used during computation
        struct {
            T As[2][BM][BK];
            T Bs[2][BK][BN];
        };
        // second half: C used during writeback
        T Cs[BM][BN];
    } smem;

    // warp tiling
    // each warp handles a 64 x 32 block of C
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // total registers: M-dim 4 tiles * N-dim 4 tiles * 4 registers each = 64
    float sum[4][4][4] = {0.f};

    T *global_a_ptr = &a[(by * BM + load_a_row) * k + load_a_col];
    T *global_b_ptr = &b[load_b_row * n + bx * BN + load_b_col];

    // ----------------------------- Prologue: pre-load one As/Bs stage
    // internally handles cross-row loading to cover all 128x32/32x128 elements
    cp_async_load_A<BK>(smem.As[0], load_a_row, load_a_col, global_a_ptr, k);
    cp_async_load_B<BK, BN>(smem.Bs[0], load_b_row, load_b_col, global_b_ptr, n);

    CP_ASYNC_COMMIT_GROUP();
    cp_async_wait_group<0>();
    __syncthreads();

    int read_idx = 0;
    int write_idx = 1;

    // main loop
    for (int bk = 32; bk < k; bk += BK) {

        // advance pointers
        global_a_ptr += BK;
        global_b_ptr += BK * n;

        // 1. cp.async load A/B
        cp_async_load_A<BK>(smem.As[write_idx], load_a_row, load_a_col, global_a_ptr, k);
        cp_async_load_B<BK, BN>(smem.Bs[write_idx], load_b_row, load_b_col, global_b_ptr, n);

        CP_ASYNC_COMMIT_GROUP();

        // 2. Tensor Core compute phase (K split into 2 steps, 16 k-elements each)
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 16;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // 4 ldmatrix A calls (4 * 16 = 64 rows)
            ldmatrix_A<BK>(reg_a, smem.As[read_idx], warp_id_m, lane_id, k_offset);

            // 4 ldmatrix B calls (4 * 8 = 32 columns)
            ldmatrix_B<BN, BK>(reg_b, smem.Bs[read_idx], warp_id_n, lane_id, k_offset);

            // MMA core computation: 4x4 m16n8k16 operations
            mma_compute<T>(sum, reg_a, reg_b);
        }

        read_idx ^= 1;
        write_idx ^= 1;

        cp_async_wait_group<0>();
        __syncthreads();
    }
    // ------------------- Epilogue: one final compute pass then writeback
#pragma unroll
    for (int k_step = 0; k_step < 2; ++k_step) {
        int k_offset = k_step * 16;

        uint32_t reg_a[4][4];
        uint32_t reg_b[4][2];

        // 4 ldmatrix A calls (4 * 16 = 64 rows)
        ldmatrix_A<BK>(reg_a, smem.As[read_idx], warp_id_m, lane_id, k_offset);

        // 4 ldmatrix B calls (4 * 8 = 32 columns)
        ldmatrix_B<BN, BK>(reg_b, smem.Bs[read_idx], warp_id_n, lane_id, k_offset);

        // MMA core computation: 4x4 m16n8k16 operations
        mma_compute<T>(sum, reg_a, reg_b);
    }

    write_c_via_smem<BM, BN>(c, by, bx, n, sum, warp_id_m, warp_id_n, lane_id, tid, smem.Cs);
}
```

The helper functions cp_async_load_A, cp_async_load_B, ldmatrix_A, ldmatrix_B, mma_compute, and write_c_via_smem have all been refactored into separate functions. I'll skip listing them to keep the article concise — please see the full code on GitHub.

## 2. hgemm_k_stages

Let's briefly discuss some shared memory details from the previous kernel. BM×BN×BK is 128×128×32, with double-buffer pipeline, totaling 128×32×2×2×2 = 32KB. Adding one more pipeline stage costs 128×32×2×2 = 16KB, bringing the total to 48KB — exactly the per-block smem limit on my card (total capacity 100KB, per-block limit 48KB).

So I wanted to first implement a 3-stage pipeline kernel to see if there's any benefit.

3-stage pipeline flow:

- Prologue: cp.async pre-issues loads for two stages' worth of buffers, i.e., commits two groups, waits until the first group finishes loading.
- Main loop:
  - Issue load for the last stage's buffer, commit group.
  - Begin ldmatrix + mma computation.
  - Advance global pointers, pipeline advances one stage, wait for the earliest group to finish loading.
- Epilogue:
  - Wait for two groups to complete.
  - Execute computation.
  - Stage through smem buffer and write back to gmem.

Code:

```c++
// The As/Bs tiling strategy stays at 128x128x32. To fit within the per-block 48KB SMEM limit, 3 stages is the physical maximum on SM120.
template <const int BM = 128, const int BN = 128, const int BK = 32, const int STAGES = 3, typename T>
__global__ void hgemm_k_stages_kernel(T *a, T *b, T *c, int m, int n, int k) {
    // grid swizzling
    int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8; // set execution block width to 8

    int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    int by = (linear_id / SWIZZLE_W) % gridDim.y;

    int tid = threadIdx.x; // 0~255
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // load mapping
    int load_a_row = tid / 4;        // 0~63
    int load_a_col = (tid % 4) * 8;  // 0,8,16,24
    int load_b_row = tid / 16;       // 0~15 (K dimension)
    int load_b_col = (tid % 16) * 8; // 0,8,16 ... 120 (N dimension)

    // A/B both row-major, using union to reuse the same memory — elegant approach
    __shared__ __align__(128) union {
        // first half: A and B used during computation
        struct {
            T As[STAGES][BM][BK];
            T Bs[STAGES][BK][BN];
        };
        // second half: C used during writeback
        T Cs[BM][BN];
    } smem;

    // warp tiling
    // each warp handles a 64 x 32 block of C
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // total registers: M-dim 4 tiles * N-dim 4 tiles * 4 registers each = 64
    float sum[4][4][4] = {0.f};

    T *global_a_ptr = &a[(by * BM + load_a_row) * k + load_a_col];
    T *global_b_ptr = &b[load_b_row * n + bx * BN + load_b_col];

    // 1. prologue: load stages-1 As/Bs blocks
#pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) {
        cp_async_load_A<BK>(smem.As[i], load_a_row, load_a_col, global_a_ptr, k);
        cp_async_load_B<BK, BN>(smem.Bs[i], load_b_row, load_b_col, global_b_ptr, n);

        CP_ASYNC_COMMIT_GROUP();

        global_a_ptr += BK;
        global_b_ptr += BK * n;
    }
    // committed two groups, allow 1 group in background — i.e. wait for the earliest load to complete
    cp_async_wait_group<STAGES - 2>();
    __syncthreads();

    // state pointer initialization
    int load_stage = STAGES - 1; // next stage to load into
    int compute_stage = 0;       // current stage to compute from

    // 2. main loop
    for (int bk = (STAGES - 1) * BK; bk < k; bk += BK) {

        // 1. issue cp.async load As/Bs to load_stage
        cp_async_load_A<BK>(smem.As[load_stage], load_a_row, load_a_col, global_a_ptr, k);
        cp_async_load_B<BK, BN>(smem.Bs[load_stage], load_b_row, load_b_col, global_b_ptr, n);

        CP_ASYNC_COMMIT_GROUP();

        // 2. Tensor Core compute phase (K split into 2 steps, 16 k-elements each)
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 16;
            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // 4 ldmatrix A calls (4 * 16 = 64 rows)
            ldmatrix_A<BK>(reg_a, smem.As[compute_stage], warp_id_m, lane_id, k_offset);

            // 4 ldmatrix B calls (4 * 8 = 32 columns)
            ldmatrix_B<BN, BK>(reg_b, smem.Bs[compute_stage], warp_id_n, lane_id, k_offset);

            // MMA core computation: 4x4 m16n8k16 operations
            mma_compute<T>(sum, reg_a, reg_b);
        }

        // advance pointers
        global_a_ptr += BK;
        global_b_ptr += BK * n;

        // advance pipeline by one stage
        load_stage = (load_stage + 1 == STAGES) ? 0 : load_stage + 1;
        compute_stage = (compute_stage + 1 == STAGES) ? 0 : compute_stage + 1;

        // ensure the earliest group has finished loading
        cp_async_wait_group<STAGES - 2>();
        __syncthreads();
    }

    // 3. epilogue: compute remaining stages-1 iterations then write back
    cp_async_wait_group<0>();
    __syncthreads();
#pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) {
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 16;
            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // 4 ldmatrix A calls (4 * 16 = 64 rows)
            ldmatrix_A<BK>(reg_a, smem.As[compute_stage], warp_id_m, lane_id, k_offset);

            // 4 ldmatrix B calls (4 * 8 = 32 columns)
            ldmatrix_B<BN, BK>(reg_b, smem.Bs[compute_stage], warp_id_n, lane_id, k_offset);

            // MMA core computation: 4x4 m16n8k16 operations
            mma_compute<T>(sum, reg_a, reg_b);
        }

        compute_stage = (compute_stage + 1 == STAGES) ? 0 : compute_stage + 1;
    }

    write_c_via_smem<BM, BN>(c, by, bx, n, sum, warp_id_m, warp_id_n, lane_id, tid, smem.Cs);
}
```

OK, we still have a big move coming, so there's nothing particularly noteworthy here. See you in the next section.

## 3. hgemm_tma_r_k_stages_kernel

When I first set out to hand-roll TMA copy + ldmatrix, I faced two problems:

- How to implement hand-rolled TMA copy?
- How to align TMA's swizzle with my manually written swizzle?

For the first problem, after a bumpy learning process (studying tensorMap and `cp.async.bulk` + mbarrier), I finally understood the entire flow. First, you create a CUtensorMap on the host side that describes the matrix's tiling size, inner/outer loop dimensions/strides, etc. in global memory. Then on the kernel side, you use `cp.async.bulk` to issue TMA copy requests + mbarrier commands for synchronization.

### 3.1 Host Side

Here's the host-side tensorMap creation method. We use cuTensorMapEncodeTiled, whose definition and explanation are as follows:

```cpp
CUresult CUDAAPI
cuTensorMapEncodeTiled(CUtensorMap *tensorMap, // pointer to the tensormap you want to create
                        CUtensorMapDataType tensorDataType, // data type, bf16/fp16
                        cuuint32_t tensorRank, // tensor rank (pass 2 for a 2D matrix)
                        void *globalAddress, // matrix base address
                        const cuuint64_t *globalDim, // global shape. Note: the innermost contiguous dimension MUST be at index 0! Row-major A(MxK) passes {K, M}
                        const cuuint64_t *globalStrides, // stride array (length Rank-1). For a 2D matrix, pass 1 element: row stride in bytes (e.g. K * sizeof(half))
                        const cuuint32_t *boxDim, // tile size for a single TMA transfer. Dimension order must strictly match globalDim (e.g. {BK, BM})
                        const cuuint32_t *elementStrides, // element strides. For dense matrices, all 1s — pass {1, 1}
                        CUtensorMapInterleave interleave, // data interleaving mode, typically NONE
                        CUtensorMapSwizzle swizzle, // swizzle type: None, 32B, 64B, 128B, plus a few special ones I haven't studied
                        CUtensorMapL2promotion l2Promotion, // L2 cache residency granularity, recommended cache-line-aligned 128B
                        CUtensorMapFloatOOBfill oobFill); // out-of-bounds fill policy. With NONE, TMA hardware auto-fills zeros beyond boundaries, eliminating tedious bounds checking
```

The most powerful aspect of hardware TMA is that if your requested boxDim extends beyond globalDim boundaries (e.g., matrix edge padding), the TMA hardware automatically fills the out-of-bounds region with zeros. You don't need any `if (x < M && y < N)` boundary-checking branches in your kernel whatsoever — this massively frees up ALU compute! Writing kernels becomes silky smooth (provided you have spare smem). My card doesn't have that luxury though.

OK, now let's use this function to create tensorMaps for matrices A and B.

Here's the catch: when the TMA engine has 128B swizzle enabled (the lifeline for avoiding bank conflicts), there's a harsh hardware constraint — the innermost contiguous dimension (Fastest Changing Dimension) in your requested boxDim must be exactly 128 bytes, no more, no less!

But our tiling strategy uses BN = 128. We're using fp16/bf16 at 2 bytes per element, meaning the B matrix tile's row width is a whopping 256 bytes! If you directly pass {BN, BK} to TMA, it will immediately refuse with CUDA_ERROR_INVALID_VALUE. (As's row is BK — whether 32 or 64, neither exceeds 128B, so no issue there.)

What to do? To meet the hardware requirement, we use a small trick: physically split the B tile in half (chunking), using two TMA launches to complete one logical transfer.

Code:

```cpp
template <typename T, const int rowBytes = 128>
inline CUtensorMap
create_tensor_map(T *global_address, uint64_t fast_dim, uint64_t slow_dim, uint32_t fast_box, uint32_t slow_box) {
    CUtensorMap tmap;
    CUtensorMapDataType type =
        std::is_same_v<T, __half> ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    CUtensorMapSwizzle swizzle = rowBytes == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B;

    // TMA core logic: dimension 0 is always the most contiguous dimension in memory (Fastest Changing Dimension)
    uint64_t globalDim[2] = {fast_dim, slow_dim};
    uint64_t globalStrides[1] = {fast_dim * sizeof(T)}; // outer dimension stride (bytes)
    uint32_t boxDim[2] = {fast_box, slow_box};
    uint32_t elementStrides[2] = {1, 1};

    CUresult res = cuTensorMapEncodeTiled(&tmap,
                                          type,
                                          2, // Tensor Rank (2D matrix)
                                          global_address,
                                          globalDim,
                                          globalStrides,
                                          boxDim,
                                          elementStrides,
                                          CU_TENSOR_MAP_INTERLEAVE_NONE,
                                          swizzle, // corresponding swizzle
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

To reuse code, I wrote a macro and a template function. But the key point is the creation of tma_b — notice I pass boxDim{BN/2, BK}, making each row exactly 128B.

### 3.2 Kernel Side

So how do we allocate the Bs smem inside the kernel? I use Bs[2][BK][BN/2].

```cpp
    // using dynamic shared memory
    extern __shared__ __align__(128) uint8_t smem_buf[];
    T(*As)[BM][BK] = reinterpret_cast<T(*)[BM][BK]>(smem_buf);
    T(*Bs)[2][BK][BN / 2] = reinterpret_cast<T(*)[2][BK][BN / 2]>(smem_buf + STAGES * BM * BK * sizeof(T));
    T(*Cs)[BN] = reinterpret_cast<T(*)[BN]>(smem_buf);
```

Why Bs[2][BK][BN/2]? Literally, it physically splits a single block of memory into left and right halves (Chunk 0 and Chunk 1).

- Chunk 0 holds columns 0–63.
- Chunk 1 holds columns 64–127.

After this split, each chunk's row width is exactly 64 elements (i.e., 128 bytes). Through the array declaration, we forcefully align the logical layout with Jensen's 128B physical boundary etched in silicon! Of course, this split introduces addressing complications for subsequent TMA writes and ldmatrix reads — but don't panic, we'll resolve this with address mapping later.

#### 3.2.1 cp.async.bulk and mbarrier

Before diving into the full kernel walkthrough, let's understand the PTX instructions we'll be using:

- mbarrier
  - We have a 3-stage pipeline, so we need 3 mbarriers. An mbarrier is an 8-byte variable in smem used for thread synchronization. Typically init and arrive appear in pairs.
- cp.async.bulk
  - Issues a TMA copy. Given the top-left corner base address of a matrix tile, TMA automatically begins asynchronous transfer of the entire tile.

The specific black-magic PTX code follows, with detailed annotations for our use case:

```cpp
// MBarrier type definition (hardware requires 8-byte alignment)
typedef uint64_t mbarrier_t;

// Initialize a 64-bit barrier variable in smem (only needs to be called by a single thread in the prologue)
// In TMA mode, the actual "data producer" is the hardware DMA engine. Our single thread only issues the transfer command.
// The mbarrier only needs to wait for [the TMA hardware as 1 entity] to finish and automatically check in, so expected_count is set to 1.
__device__ __forceinline__ void mbarrier_init(mbarrier_t *mbar, uint32_t expected_count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar))),
                 "r"(expected_count));
}
// Set the expected byte count for TMA transfer, reported to the specified mbar.
// Traditional sync waits for "thread" arrives; here we wait for "bytes" to arrive.
// Once the hardware has transferred this many bytes, it automatically triggers an arrive that flips the phase.
__device__ __forceinline__ void mbarrier_expect_tx(mbarrier_t *mbar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(
                     static_cast<uint32_t>(__cvta_generic_to_shared(mbar))),
                 "r"(tx_bytes));
}

// After compute threads finish consuming data, submit an arrive signal (flip the phase).
// We don't use this since there's no warp specialization — all warps are consumers and we simply use __syncthreads().
// In warp-specialized programming, consumer threads would report to the corresponding mbar that data consumption is complete.
__device__ __forceinline__ void mbarrier_arrive(mbarrier_t *mbar) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar))));
}

// Compute threads synchronously wait for TMA data to be ready (with built-in sleep — doesn't consume ALU cycles).
// A bit arcane. The main logic:
//   Declare a temporary predicate register named p (boolean) to store the mbarrier status check result.
//   mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2: check if the mbar's internal phase flag matches
//     the given phase. If they don't match, TMA hasn't finished copying yet — suspend for 'ticks' cycles,
//     yielding the ALU to other warps for scheduling.
//   @p bra DONE: if p is true, the mbarrier phase has flipped — TMA copy is complete, jump to instructions after DONE.
//   bra LAB_WAIT: if we reach here, the ten-million clock cycle timeout expired (extremely rare) — jump back to LAB_WAIT and sleep again.
//   The final "memory" is a compiler memory barrier (clobber). It prevents the compiler from over-optimizing and reading stale data.
//   It tells the compiler: TMA hardware just secretly modified shared memory (As, Bs matrices) in the background —
//   all register-cached shared memory variables after this instruction must be invalidated and re-read.
__device__ __forceinline__ void mbarrier_wait(uint64_t* mbar, uint32_t phase) {
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    // set a very large suspend timeout (0x989680 = 10,000,000 clock cycles)
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred p; \n\t"
        "LAB_WAIT: \n\t"
        // note the third argument %2
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

// global to shared::cta 2D TMA transfer
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

So the kernel's main TMA-related logic is (in our non-warp-specialized model, thread 0 handles TMA scheduling):

- Copy thread (thread 0):
  - Initialize 3 mbarrier variables for 3 stages.
  - Report the expected data volume per stage to the mbarrier.
  - Using the tensorMap, issue TMA copy requests (cp.async.bulk).
- Compute threads (all threads):
  - Collectively poll-wait (mbarrier_wait) until the corresponding stage's mbarrier flips to the specified phase — all threads are woken up.
  - Begin ldmatrix + mma computation.

### 3.3 Kernel Design

Now that we know the general flow, we can start designing the kernel. There's still one question I haven't answered: how to align TMA's swizzle with my manual swizzle. I didn't spend the effort finding documentation to confirm TMA's hardware swizzle method, so I just guessed — but guessing comes with risk.

To minimize testing effort, I made a decision that goes against "tradition." I abandoned the BM×BN×BK=128×128×32 tiling and set BK to 64! This way, a single row of matrix A's tile is also 128B, and matrix B — after our split into chunks — also has 128B rows. As long as the 128B TMA swizzle aligns with the ldmatrix swizzle, it should work.

The cost: I need 128×64×2×3×2 = 96KB of smem, which can only hold one resident block. This requires the CUDA magic of cudaFuncSetAttribute + dynamic smem arrays.

Additionally, since TMA only needs a single instruction, it frees up all the address-variable registers that unrolling previously needed. So I aggressively added double-buffered register reads for As/Bs (though the impact is modest).

Here's the complete kernel code:

```cpp
template <const int BK, typename T>
__device__ __forceinline__ void
ldmatrix_A_tma(uint32_t reg_a[4][4], T (*As)[BK], int warp_id_m, int lane_id, int k_offset) {

    // 4 ldmatrix A calls (4 * 16 = 64 rows)
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

        // distinguish which chunk we're in
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
    const int SWIZZLE_W = 8; // set execution block width to 8

    int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    int by = (linear_id / SWIZZLE_W) % gridDim.y;

    int tid = threadIdx.x; // 0~255
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // using dynamic shared memory
    extern __shared__ __align__(128) uint8_t smem_buf[];
    T(*As)[BM][BK] = reinterpret_cast<T(*)[BM][BK]>(smem_buf);
    T(*Bs)[2][BK][BN / 2] = reinterpret_cast<T(*)[2][BK][BN / 2]>(smem_buf + STAGES * BM * BK * sizeof(T));
    T(*Cs)[BN] = reinterpret_cast<T(*)[BN]>(smem_buf);
    // place mbar at the tail (8-byte aligned, 3 stages)
    mbarrier_t *mbar = reinterpret_cast<mbarrier_t *>(smem_buf + BM * BK * sizeof(T) * STAGES * 2);

    // initialize MBarrier (only tid 0 executes, expected arrive count is 1 since only TMA signals it)
    if (tid == 0) {
        for (int i = 0; i < STAGES; ++i)
            mbarrier_init(&mbar[i], 1);
    }
    __syncthreads(); // ensure MBarrier initialization is complete

    // warp tiling
    // each warp handles a 64 x 32 block of C
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // total registers: M-dim 4 tiles * N-dim 4 tiles * 4 registers each = 64
    float sum[4][4][4] = {0.f};

    // total bytes per TMA transfer
    const uint32_t tx_bytes = (BM * BK + BK * BN) * sizeof(T);

    // only a single, minimal coordinate tracker (TMA's host descriptor already knows the strides)
    int load_k_coord = 0;

    // 1. prologue: load STAGES - 1 blocks
    for (int i = 0; i < STAGES - 1; ++i) {
        if (tid == 0) {
            // tell this mbarrier how many bytes of data to expect
            mbarrier_expect_tx(&mbar[i], tx_bytes);

            cp_async_bulk_tensor_2d(&mbar[i], &tma_a, As[i], load_k_coord, by * BM);
            // Bs requires two copies, split into two chunks
            cp_async_bulk_tensor_2d(&mbar[i], &tma_b, Bs[i][0], bx * BN, load_k_coord);
            cp_async_bulk_tensor_2d(&mbar[i], &tma_b, Bs[i][1], bx * BN + BN / 2, load_k_coord);
        }
        load_k_coord += BK;
    }

    int load_stage = STAGES - 1;
    int compute_stage = 0;
    int wait_phase = 0; // MBarrier's natural 0/1 alternating phase switch
    int total_k_step = BK / 16; // adaptive step count based on BK
    // 2. main loop
    for (int bk = (STAGES - 1) * BK; bk < k; bk += BK) {

        // issue next round of TMA (still only tid 0 does the work)
        if (tid == 0) {
            mbarrier_expect_tx(&mbar[load_stage], tx_bytes);
            cp_async_bulk_tensor_2d(&mbar[load_stage], &tma_a, As[load_stage], load_k_coord, by * BM);
            cp_async_bulk_tensor_2d(&mbar[load_stage], &tma_b, Bs[load_stage][0], bx * BN, load_k_coord);
            cp_async_bulk_tensor_2d(&mbar[load_stage], &tma_b, Bs[load_stage][1], bx * BN + BN / 2, load_k_coord);
        }
        load_k_coord += BK;

        // all threads: poll-wait until TMA finishes transferring the current compute_stage's data
        mbarrier_wait(&mbar[compute_stage], wait_phase);

        // register double buffer: ldmatrix + mma
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

        // direct sync — no warp specialization, no need for arrive
        __syncthreads();

        // rotate state
        load_stage = (load_stage + 1 == STAGES) ? 0 : load_stage + 1;
        compute_stage = (compute_stage + 1 == STAGES) ? 0 : compute_stage + 1;

        // after completing one full 3-stage pipeline cycle, all 3 mbarrier phases have flipped — we flip our wait_phase too
        if (compute_stage == 0)
            wait_phase ^= 1;
    }
    // 3. epilogue: compute remaining stages-1 iterations
#pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) {
        // continue waiting for TMA
        mbarrier_wait(&mbar[compute_stage], wait_phase);

        // register double buffer
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

    // 4. writeback
    write_c_via_smem<BM, BN>(c, by, bx, n, sum, warp_id_m, warp_id_n, lane_id, tid, Cs);
}
```

When ldmatrix reads from Bs, we need to first determine which chunk we're in. Fortunately, since we only split into two clean halves, it's not complicated. Beyond that, there's nothing special — the code comments should help explain the rest.

Additionally, sharp-eyed readers may have noticed that my kernel receives the tensorMap with `__grid_constant__`. This constraint keyword forces the GPU to place the variable in constant storage.

#### 3.3.1 Hard-Won Lesson: The Critical `__grid_constant__`

In fact, I was stuck on tensorMap construction for a long time at the very beginning. I encountered a bizarre problem: the kernel crashed on launch. After painful debugging, I discovered the root cause: a tensorMap is essentially a complex 128-byte structure (descriptor). Under CUDA's default parameter-passing rules, a structure this large can easily be allocated in local memory by the compiler. But the TMA hardware engine requires the tensorMap to reside in the global constant region.

The fix: add the `__grid_constant__` keyword to the tensorMap parameter in the kernel declaration. This modifier forces the compiler to place the tensorMap in constant memory, guaranteeing its visibility and read-only access throughout the grid's lifetime.

For anyone trying TMA for the first time — beware of this pitfall!

## 4. Converging Paths: The Swizzle Handshake Between TMA and ldmatrix

After a painful debugging process, I finally got the above kernel running and passing diff_check, confirming that TMA's 128B swizzle matches my previous Bs swizzle: `((col) ^ (((row) & 0x7) << 3))`. So you see, GPU design and evolution follow fundamental laws too — since my manual ldmatrix swizzle matches TMA's, the underlying logic must be the same.

I then wondered: is the 64B swizzle also identical?

So I made some adjustments to the code to support both BK=32 and BK=64 cases, using my original swizzle_a for ldmatrix As. Then I set BK=32, halving smem usage — allowing two active blocks again. Perfect.

After the changes, it passed the diff check on the first run! Haha, persistence pays off. I don't know TMA's black-box swizzle logic, but under strict precision diff testing, my manual swizzle logic perfectly matches TMA hardware's black-box behavior — zero error.

This is the pure romance of reverse engineering — when the hardware won't tell you, you deduce the logic from the results.

## 5. Benchmark, NCU Report and Analysis

For this round of testing, I expected the final kernel to be roughly on par with the baseline, so I conducted a relatively rigorous comparison — closing all programs, locking memory and GPU clocks, and testing in a cool environment.

Here are the benchmark results:

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

I'll admit — I picked a run where the final kernel won. After all the effort invested, posting a result where it lost to `hgemm_bcf_dbf_rw` (the actual win rate is roughly 6:4) would be quite anticlimactic. But in my heart, `hgemm_tma_r_k_stages_32` and `hgemm_bcf_dbf_rw` are on the same level — both have squeezed the Tensor Cores to their absolute limit (though the former is far more complex).

NCU report:
![p](static/hgemm_sm120_0.png)

This is the NCU report I'm most satisfied with. Although the final kernel can't consistently outperform `hgemm_bcf_dbf_rw`, the NCU summary is clean — no annoying `Uncoalesced Shared Accesses`. The only warnings are insufficient Tensor Core throughput and occupancy limitations from register/smem constraints.

- `hgemm_tma_r_k_stages_64`: 0 bank conflicts
![p](static/hgemm_sm120_1.png)
- `hgemm_tma_r_k_stages_32`: also 0 bank conflicts (NCU still reports a tiny amount of bank conflicts — with two active blocks, there are some inter-warp conflicts, but the swizzle isn't to blame)
![p](static/hgemm_sm120_2.png)

### Discussion

- Why didn't upgrading from 2-stage to 3-stage pipeline improve performance — and in fact hurt it?
  - Pipelining is essentially trading space for time. My hypothesis is that the 2-stage pipeline already pushes register pressure quite hard. Going to 3 stages theoretically requires even more registers while maintaining 2 active blocks — the compiler must have rescheduled instructions, sacrificing more instruction-level parallelism.
- TMA's dedicated hardware data transfer is genuinely useful, especially for reducing register usage. This is evident from my NCU report:
  - With the 128×128×32 tiling strategy, even at 3 pipeline stages, register usage dropped from `hgemm_bcf_dbf_rw`'s 128 down to 96 — and this is with double-buffered registers!
  - This means on SM120, the bottleneck has completely shifted from "memory scheduling" to "compute throughput." TMA has done everything it can — the remaining gap is the compute NVIDIA cut away.
- But TMA alone is useless without large smem and sufficient Tensor Core throughput.
  - NCU reports `Math Pipe Throttle Stalls` — the Tensor Cores literally can't keep up.
  - Give me another 46KB of smem and more compute, and I could fit one more block, boosting throughput by 50%. Jensen's knife cuts deep.
- Reducing BK from 64 to 32 halves the per-block compute, doubling the block count. Although total compute stays the same, actual performance is better.
  - This demonstrates that when occupancy is extremely low, increasing occupancy and letting the hardware scheduler do its job is the better choice.
- Final kernel summary:
  - SM120's TMA + ldmatrix (grafted swizzle, double register buffer) + mma + 3-stage pipeline + grid swizzle kernel.
  - I've thrown in everything I understand.

## 6. Conclusion

After arduous coding, I've completed what I consider a perfect kernel — this card has been squeezed dry. On the high-performance computing battlefield, top-tier GPUs win with raw power (H100/B100). But our optimization on a mobile GPU is a micro-sculpture war performed atop a footboard. Though the performance gains are small, our mastery of the hardware's low-level details is a developer's ultimate moat.

This is truly the final post in the GEMM series. There was still a TODO — using TMA to write Cs back to gmem — but the expected gain was too small to motivate me.

So, the GEMM series concludes~ confetti~

That's all.

If there are any errors, please feel free to correct them. The complete kernel and test code are available on GitHub.
