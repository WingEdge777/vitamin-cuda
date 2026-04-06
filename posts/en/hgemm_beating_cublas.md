# [CUDA in Practice] HGEMM — Beating cuBLAS: Tensor Core, cp.async, ldmatrix, mma
>
>Note: Text translated by AI. Code crafted by human.
>

## 0. Preface — Half Precision Rules the World
>
> This article is intended for readers with a solid foundation in CUDA programming who are familiar with GEMM optimization and interested in advanced Tensor Core / inline PTX instruction performance tuning.
>
> The complete kernel and test code can be found at [hgemm](/kernels/hgemm).
>
> That's right — this is the third installment of the "Beating cuBLAS" series. (As long as NVIDIA doesn't bother tuning for mobile GPUs, we'll keep beating them. This is likely the final GEMM post though — I might write an fp8 kernel, but probably not a pure-blooded fp8 GEMM. It's starting to feel a bit samey, and fp8 really needs to be paired with various quantization schemes anyway.)
>

Nowadays bf16 and even fp8 dominate LLM/VLM training and inference. Traditional models may still use fp16 (slightly higher precision but narrower dynamic range) — their weight and activation distributions are more concentrated with fewer outliers, making fp16 a better fit. LLMs and VLMs are generally trained in bf16, with inference potentially using fp8/int8/int4 and other quantization schemes. So half-precision matrix multiplication is the true battleground.

This article uses M=N=K=4096 (MxKxN, the medium-scale sweet spot where cuBLAS excels) GEMM in fp16/bf16 as an example. Running on an RTX 5060 Mobile GPU, using PTX instructions like cp.async, ldmatrix, and mma alongside Tensor Core acceleration, we successfully outperformed NVIDIA's own cuBLAS in same-precision computation. This article walks through the entire journey of competing against cuBLAS — covering cp.async instructions, ldmatrix/mma warp-level PTX usage, hardcore swizzle derivation, layout analysis and intuition, and more.

We present 4 kernel implementations, progressing from basic grid swizzle + cp.async + dual ldmatrix + mma, to swizzled smem read/write for bank conflict resolution, double buffering for latency hiding, and finally coalesced gmem read/write transactions (yes, filling the hole from the previous article about C matrix writeback transaction coalescing) — ultimately beating cuBLAS. The process covers instruction requirements/usage, swizzle design, and aims to help readers deeply understand Tensor Core usage and optimization techniques.

Kernel outline (the first one is the cuBLAS kernel):

- hgemm_cublas bf16/fp16 version
- hgemm_naive bf16/fp16 version (ldmatrix + mma)
- hgemm_bcf bf16/fp16 version (ldmatrix + mma, As/Bs swizzle bcf, 95–99% of cuBLAS performance)
- hgemm_bcf_dbf bf16/fp16 version (ldmatrix + mma, As/Bs swizzle bcf, double buffer, outperforming cuBLAS)
- hgemm_bcf_dbf_rw bf16/fp16 version (ldmatrix + mma, As/Bs swizzle bcf, double buffer, coalesced r/w gmem, outperforming cuBLAS)

## 1. hgemm_naive

We've already covered the basic usage of cp.async, ldmatrix, and mma in previous articles. Here's a brief recap — only the specific instruction forms used, omitting input/output operand lists. For full details, refer to the official documentation.

```cpp
cp.async.cg.shared.global.L2::128B [%0], [%1], 16; // cg: (Cache at Global level): bypass L1, copy 16 bytes, prefetch 128B to L2
cp.async.commit_group;  // commit the async copy group
cp.async.wait_group 0;  // number of async groups allowed in the background; 0 means wait for all to complete
```

- `cp.async`: Usage is identical to the tf32 kernel — no changes whatsoever.

```cpp
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32        d, a, b, c;
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32        d, a, b, c;
```

- `mma` instruction: slight change.
  - Previous compute shape was m16n8k8 (a.k.a. 1688), now it's m16n8k16.
  - Previous a/b types were tf32, now switched to f16/bf16.

```cpp
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];
ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];
```

- ldmatrix: A warp cooperatively loads a small matrix tile from smem into all threads' registers. The result held across all threads' registers is called a fragment (the output held by all threads is called fragment C).
  - Note: x2 and x4 indicate 16 and 32 address-providing threads respectively. Each address-providing thread in ldmatrix always loads exactly 16 bytes! After loading, data is distributed across threads to form a fragment.
    - Understanding this is key to understanding how to resolve bank conflicts.
  - Since we're using the m16n8k16 mma shape for half precision, we need to load a 16x16 tile from As and a 16x8 tile from Bs.
  - For half precision, ldmatrix now supports `.trans` transpose (hooray~).

Building on our tf32 experience, let's briefly mention the tiling strategy, then jump straight to the code.

- 128x128 tiling (from C matrix's perspective), K-dimension stride of 32 (since half precision is 2 bytes, we can aggressively load 32 k-elements), 256-thread block size.
- A 2D 2x4 warp tiling, dividing the C tile into upper and lower 64x128 halves — each warp handles a 64x32 block. This maps to m16n8k16 with 4x4 iterations (k-dimension counted separately).
- The tiling size choice involves many considerations (please refer to my first SGEMM article for details, thank you).

Code:

```cpp
// a block calculate c[128][128]
template <const int BM = 128, const int BN = 128, const int BK = 32, typename T>
__global__ void hgemm_naive_kernel(T *a, T *b, T *c, int m, int n, int k) {
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

    // A/B both row-major
    __shared__ T As[BM][BK];
    __shared__ T Bs[BK][BN];

    // warp tiling
    // each warp handles a 64 x 32 block of C
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // total registers: M-dim 4 tiles * N-dim 4 tiles * 4 registers each = 64
    float sum[4][4][4] = {0.f};

    // main loop
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

        // 3. Tensor Core compute phase (K split into 2 steps, 16 k-elements each)
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 16;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // 4 ldmatrix A calls (4 * 16 = 64 rows)
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
                // ldmatrix x4 loads 16x16
                int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
                int a_col = k_offset + (lane_id / 16) * 8;
                uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&As[a_row][a_col]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

            // 4 ldmatrix B calls (4 * 8 = 32 columns)
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                // Lanes 0~15 cover exactly 16 rows (start addresses of two 8x8 blocks)
                int b_row = k_offset + (lane_id % 16);
                int b_col = warp_id_n * 32 + n_idx * 8;

                uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[b_row][b_col]));
                LDMATRIX_X2_TRANS(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
            }

            // MMA core computation: 4x4 m16n8k16 operations
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

    // ---------------- Write back C matrix ----------------
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

Obviously, the naive version doesn't apply swizzle to As/Bs reads, so there are definitely bank conflicts. Let's optimize shared memory access.

In the previous tf32 article, we already had two swizzle macro functions:

```c++
#define SWIZZLE_A(row, col) ((col) ^ (((row >> 1) & 0x3) << 2))

#define SWIZZLE_B(row, col) ((col) ^ (((row) & 0x7) << 3))
```

Can we directly reuse these swizzle functions in the half-precision kernel? Obviously not — because we're now working with half-precision data. The As and Bs layouts are 128x32 and 32x128, meaning As's col index ranges from 0–31 and so does Bs's row index.

But using these as a starting point, deriving the half-precision swizzle from tf32 is actually quite straightforward. Think about it: half precision is 2 bytes while tf32 was 4 bytes. This means the column index that was previously 16-byte-aligned now doubles in range. In other words, where we previously only needed to preserve bits 0–1 of col from disturbance, we now need to preserve bits 0–2. So SWIZZLE_B already satisfies the requirement and can be used directly, but SWIZZLE_A needs modification — a simple multiply by 2.

Thus, we arrive at the two conflict-free swizzle functions for As/Bs:

```cpp
#define SWIZZLE_A(row, col) ((col) ^ (((row >> 1) & 0x3) << 3))

#define SWIZZLE_B(row, col) ((col) ^ (((row) & 0x7) << 3))
```

Core code changes:

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

// reads
...
uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&As[a_row][SWIZZLE_A(a_row, a_col)]));

...
uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[b_row][SWIZZLE_B(b_row, b_col)]));
```

## 3. hgemm_bcf_dbf

The previous version resolved bank conflicts while maintaining 16-byte alignment. Now we simply add double-buffer pipelining (see my first article for the detailed pipeline flow). Code omitted here — the next section covers the key remaining optimization.

## 4. hgemm_bcf_dbf_rw

In our previous tf32 kernel, we noted that the C matrix writeback to global memory wasn't using coalesced transactions. It's actually doable — we were just being lazy. Now that all the prerequisite steps are sorted out and the above wasn't too brain-intensive, let's fill this gap.

The core idea is simple: repurpose the As/Bs shared memory as an intermediate buffer, have all threads write fragment C results into smem in a well-organized layout, then use float4 vectorized instructions to read from smem and blast everything back to gmem in one shot.

Here's a beautifully convenient coincidence: the shared memory occupied by As and Bs is 2 × (128×32 + 32×128) × 2B = 128×128×2 — this is exactly enough to hold the 128x128 C matrix tile computed by one block! If it didn't fit and required batched staging, I probably wouldn't have bothered writing this (laughs).

Before we start, we need to thoroughly understand the register state of fragment C. After the mma.sync.m16n8k16 instruction computes, it outputs a 16x8 sub-matrix. We need to understand exactly which values in this matrix each thread holds. The official NVIDIA documentation has a diagram — for the m16n8k16 compute shape, fragment C is:

![](static/fragment_c.png)

For a more intuitive view, I drew my own diagram:

![](static/my_fragment_c.svg)

The pattern is quite clear: within a warp, every 4 threads are responsible for 8 elements (fp16/bf16) in the same row. After skipping 8 rows, this arrangement repeats. At the per-thread level, each thread holds 4 values. For example:

- T0 holds c[0][0], c[0][1] and c[8][0], c[8][1] (8 rows below)
- T1 holds c[0][2], c[0][3], c[8][2], c[8][3]
- ... and so on

Of course, don't forget there's an outer 4(m)×4(n) loop. This means in the C matrix's global coordinate system, each iteration strides by 16 rows or 8 columns.

With the register distribution understood, the task is: stitch together the data held by T0–T31, row-aligned and column-aligned, and write it into our 128x128 smem buffer.

But — bank conflict alert incoming!

Within a warp, every 4 threads write one row, covering 8 rows total. In our 128-wide smem, the same column across different rows is perfectly aligned in physical addresses. If we write directly, threads from all 8 rows will instantly collide on the same bank, causing catastrophic 8-way bank conflicts!

However, staggering the banks is easy — we can simply reuse the B matrix's swizzle macro. Why? Taking warp 0 with m/n offsets both equal to 0 as an example:

```yaml
row: 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7
col: 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6
```

![](static/swizzle_c.svg)

Looking at the binary closely: row is 0–7, i.e., 00xxx — the effective variable bits are bit0–2. Since we also need to maintain 16-byte alignment, we take the low 3 bits and left-shift by 3 positions, which perfectly stagger against col's low 3 bits. After XOR, this covers all 32 banks.

> Wait — someone might say, shifting left by 3 lands on bit5, which exceeds 32 — that doesn't seem right.

Pay attention: since we're indexing fp16/bf16 elements (2 bytes each), the bank ID formula is (row×128 + col)×2 / 4 % 32 = (col/2) % 32. Here we actually need to consider 6 bits, or equivalently, bit5 gets shifted back to bit4 — gotcha~)

Other warps with different offsets follow the same logic. With this elegant bit manipulation, we write the C matrix into shared memory with zero conflicts.

Core changes:

```cpp
#define SWIZZLE_C(row, col) ((col) ^ (((row) & 0x7) << 3))

...
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

...

// reuse As/Bs for staging
__syncthreads();

int t_row = lane_id / 4;       // 0~7
int t_col = (lane_id % 4) * 2; // 0, 2, 4, 6 — every 4 threads handle 8 columns per row = 16 bytes

// register to Cs smem
#pragma unroll
for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
    for (int n_idx = 0; n_idx < 4; ++n_idx) {
        int c_base_row = warp_id_m * 64 + m_idx * 16; // m strides 16 rows
        int c_base_col = warp_id_n * 32 + n_idx * 8;  // n strides 8 columns

        // split 16 rows into two batches of 8 rows each
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
// each thread moves 64 elements (fp16/bf16), i.e. 8 float4s; 256 threads write 256*4*4 = 4096 bytes per step
T *c_block = &c[by * BM * n + bx * BN];

#pragma unroll
for (int step = 0; step < 8; ++step) {
    // ensure 32 threads in the same warp have absolutely contiguous elem_idx at this point
    int elem_idx = (step * 256 + tid) * 8;
    int row = elem_idx / 128;
    int col = elem_idx % 128;

    int s_col = SWIZZLE_C(row, col);

    FLOAT4(c_block[row * n + col]) = FLOAT4(smem.Cs[row][s_col]);
}
```

Note: The reason we can use FLOAT4 cast for writeback is precisely because our derived `SWIZZLE_C` preserves 16-byte alignment.

## 5. Benchmark, NCU Report and Analysis

Without further ado, here are the benchmark results and NCU report:

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

### Discussion

- Per tradition, let's look at the cuBLAS kernel: `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x64_32x4_nn_align8>(T1::Params)`
  - 256x64 tiling (m=256, n=64), K-dimension split of 32, 4-stage pipeline.
- My NCU summary also flags `Uncoalesced Shared Accesses`. The specific reason is that after cp.async + swizzle, the source/destination address continuity/monotonicity no longer meets the hardware's requirements, causing cp.async to replay wavefronts for repeated writes.
  - I initially really wanted to optimize this away, hoping to write a perfect kernel. At first I didn't understand why it was uncoalesced, so I tried reworking the swizzle mapping — no luck. Then I tried moving the swizzle to the global memory read side, writing flat to smem and swizzle-reading out. Still couldn't solve it.
  - After researching and grilling Gemini: when 32 threads in a warp execute cp.async to move 512B, ideally the LSU would pack them into 4 perfect 128B memory transactions (8 threads each perfectly filling one smem row). But because of my swizzle, the originally contiguous write addresses for those 8 threads get scattered. These scattered addresses are neither contiguous nor monotonically increasing, so the crossbar can't route the data to the corresponding banks in a single clock cycle — it has to decompose the transaction, triggering multiple wavefront writes.
    - Note: I'm not entirely sure this description of the underlying microarchitectural behavior is correct and rigorous. Experts are welcome to correct and supplement in the comments — thank you.
- Why doesn't cuBLAS have this problem?
  - I don't know cuBLAS's exact implementation, but it certainly has its own costs. Open its shared memory statistics and you'll find over 500,000 bank conflicts! Meanwhile, I have 0 conflicts (actually ~0.14% from inter-warp conflicts, negligible).
  - This is a classic trade-off in extreme performance tuning: I chose absolutely conflict-free reads at the cost of losing cp.async write coalescing; cuBLAS guarantees blazing-fast async copy writes but tolerates partial conflicts during compute reads.
  - cuBLAS and I chose different compromise directions — which is better? I'll leave that for readers to discuss.

## 6. Conclusion

To summarize the techniques used: cp.async, ldmatrix, and mma PTX instructions paired with Tensor Core acceleration, grid swizzle to maximize L2 utilization, swizzle to eliminate bank conflicts, double-buffer async pipelining to hide latency, and repurposing As/Bs as staging for coalesced C matrix writeback transactions. Successfully beating cuBLAS.

This should be the final post in the pure-blooded GEMM series — from fp32 to tf32 to half precision, from naive to beating cuBLAS. Through this entire journey, I believe readers have gained a deeper understanding of GPU architecture, CUDA programming, and performance optimization.

If there are any errors, please feel free to correct them. The complete kernel and test code can be found at [hgemm](/kernels/hgemm).

That's all.
