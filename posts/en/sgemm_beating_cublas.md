# [CUDA in Practice] SGEMM — Beating cuBLAS: A Deep Dive into Peak-Performance Matrix Multiplication in Pure CUDA C++
>
>Note: Text translated by AI. Code crafted by human.
>
## 0. Preface — The Last Stand of Scalar Compute

> **Warning:** Extremely dense content ahead, with many diagrams, heavy bit-manipulation, and memory-mapping derivations. Best read on a PC.
>
> **Target audience:** Readers with a solid CUDA programming foundation, who understand basic matrix multiplication and are interested in hand-tuning kernel performance.
>
> AThe complete source code for all kernels and performance benchmarks discussed in this post is available in my open-source practice project: [WingEdge777/Vitamin-CUDA](https://github.com/WingEdge777/Vitamin-CUDA)
>
> A bit of clickbait in the title, admittedly. cuBLAS is a general-purpose, numerically precise library — the collective work of many NVIDIA experts. I obviously can't beat cuBLAS across the board. But in specific scenarios — without boundary checks, etc. — hand-crafting a kernel that surpasses cuBLAS is the mark of a competent HPC engineer. (Consider beating cuBLAS in pure C++ without intrinsics the ultimate rite of passage for an HPC engineer. :))

In an era dominated by Tensor Cores, is there still value in writing a pure FP32 SIMT scalar matrix multiply (SGEMM)? Yes. Because it remains the ultimate litmus test for a low-level compute engineer's mastery of GPU memory control, warp scheduling, shared memory / register allocation, and instruction-level parallelism (ILP).

This article takes an M=N=K=4096 (M×K×N — the mid-size regime where cuBLAS excels) matrix multiplication as the running example. On an RTX 5060 Laptop GPU, without calling any assembly-level instructions, without `cp.async` or other modern architectural features — using **pure C++ CUDA only** — I squeezed the runtime from PyTorch's 16.59 ms down to 13.85 ms, **beating NVIDIA's own cuBLAS (14.33 ms) at equal precision**. This post is a detailed retrospective of the entire battle against the `nvcc` compiler and the physical hardware.

The article presents **6 kernel implementations**, progressing from basic tiling and shared memory usage, through swizzle-based bank conflict elimination, double buffering for latency hiding, fully coalesced global memory access, ILP improvements, and register pressure management to maintain occupancy — ultimately surpassing cuBLAS. Throughout, you'll encounter intricate index mappings for data movement and computation, the philosophy of latency hiding, and be pushed to deepen your understanding of the hardware. Here's the kernel roadmap — let's dive in.

- `sgemm_naive` — no optimizations
- `sgemm_tiling` — vectorized r/w + block tiling with shared memory
- `sgemm_at_tiling` — vectorized r/w + A-transpose into SMEM (4-way write bank conflict, inner-loop `float4` reads)
- `sgemm_at_bcf_swizzling` — vectorized r/w + A-transpose + swizzle, bank-conflict-free
- `sgemm_at_bcf_swizzling_rw` — + coalesced C write-back
- `sgemm_at_bcf_swizzling_dbf_rw` — + double buffer pipeline, **outperforms cuBLAS**

## 1. Naive Implementation

Not much to say here. The naive kernel is just the textbook triple loop on the GPU: each thread computes one element of C by accumulating along K. No optimizations whatsoever.

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

## 2. Tiling + Shared Memory

No point starting from scratch in 2026. Here's a reasonably strong baseline with 2D data tiling + 2D block/thread tiling + shared memory, which we'll iteratively improve. First, based on my GPU specs (RTX 5060 Laptop) and resource limits (max SMEM 100 KB, 48 KB per block; max 65,536 registers per block), I determined the data tile size and thread block size.

![ab_tiling](../static/tiling.svg)

- From C's perspective: data tile BM×BN = 128×128, block size = 256 threads.
  - Each block computes a 128×128 sub-matrix of C. 256 threads form 8 warps in a 2×4 layout.
    - Each warp handles C[64][32]; each thread computes an 8×8 sub-tile of C.
  - K-dimension stride BK = 16. Each block loads 128×16 of A and 16×128 of B per iteration.
    - 16 is the K-dimension slice; the kernel loops along K accumulating partial products to produce the final 128×128 result.

You might ask: how did I arrive at these specific tile and block sizes? There are many factors to consider. Here are the key ones. First, know your hardware (go back to my first article and memorize the `deviceQuery` output).

**Thread block size:**

- I always choose from 128 or 256 — these are the sweet spots for typical GPUs.
- There's a compute-to-memory ratio consideration (detailed below), which makes me lean toward 256: more threads can load more data into SMEM for reuse during C computation.

**Data tile size:**

- Allocate threads from C's perspective. With 256 threads, how many rows/columns?
- Must be a multiple of 4 for vectorized access.
- Compute-to-memory ratio: my GPU's peak CUDA Core throughput is 3328 × 1455 × 1e6 × 2 ≈ 9.6 TFLOPS; peak bandwidth is 12001 × 1e6 × 128/8 × 2 ≈ 384 GB/s. Ratio: 9686/384 ≈ 25.2 FLOPs/Byte.
  - To cross the memory wall and keep the CUDA Cores saturated, each data load should yield at least 25 FMA operations. After testing a few tile sizes, I chose 128×16×128:
    - 128 × 16 × 128 × 2 / (128 × 16 × 2) / 4 = 32 > 25.2

In the end, intuition plays a non-trivial role too — I instinctively picked 128×128 square tiles, then verified the resource usage. Assuming K-stride = 16:

- **SMEM usage:** Loading 128×16 of A and 16×128 of B = 128 × 16 × 2 × 4 bytes = 16 KB per block. Well within SMEM capacity; divisible by 256 for clean thread assignment; allows 4 active blocks. Even with double buffering (32 KB), 2–3 active blocks are feasible.
- **Register usage:** Each thread needs 8×8 = 64 registers for C accumulators; plus 8×2 = 16 for A/B staging; 64 + 16 = 80. With double buffering, add another 16 → 96. Plus temporaries, address offsets, etc. — estimate 10–20+. Total across 256 threads stays well under the 65,536 limit (65536/256 = 256 per thread). At least 2 active blocks in the ideal case. Exceeding 128 registers per thread would drop to 1 active block (in fact, my initial double-buffer code did exceed 128, causing occupancy to tank — I recovered by eliminating unnecessary variables).

If BK were 8, the compute-to-memory ratio drops below the memory wall. If 32, SMEM and register usage explode, occupancy drops, and double buffering becomes impossible.

These are my personal considerations for block and tile sizing. Other valid configurations certainly exist, but mine satisfies all constraints, so I went with it. On different hardware or data scales, optimal parameters may differ entirely. In production, you'd typically auto-tune block sizes via empirical benchmarking.

With that out of the way, here's the core code:

```cpp
// a block calculate c[128][128], each thread c[8][8]
template <const int BM = 128, const int BN = 128, const int BK = 16, const int TM = 8, const int TN = 8>
__global__ void sgemm_tiling_kernel(float *a, float *b, float *c, int m, int n, int k) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x; // 0~255; 8 warps, 2x4 tiling
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // Each block loads 64x16 of A and 8x128 of B per pass, twice to cover 128x16 and 16x128.
    // Every 4 threads handle one row of A (16 elements); every 32 threads handle one row of B (128 elements).
    int load_a_row = tid / 4;               // 0~63
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12...
    int load_b_row = tid / WARP_SIZE;       // 0~8
    int load_b_col = (tid % WARP_SIZE) * 4; // 0,4,8,12,16,20,24,28...

    // Warp tiling: every 4 warps cover the upper/lower 64x128 halves of C
    int warp_row = warp_id / 4;      // 0, 1
    int warp_col = warp_id % 4;      // 0, 1, 2, 3
    int t_row_in_warp = lane_id / 4; // 0~7
    int t_col_in_warp = lane_id % 4; // 0~3

    // C output base coordinates; each thread handles an 8x8 tile
    int c_row = warp_row * 64 + t_row_in_warp * 8;
    int c_col = warp_col * 32 + t_col_in_warp * 8;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float sum[TM][TN] = {0.f};

    for (int bk = 0; bk < k; bk += BK) {
        FLOAT4(As[load_a_row][load_a_col]) = FLOAT4(a[(by * BM + load_a_row) * k + bk + load_a_col]);
        FLOAT4(As[load_a_row + 64][load_a_col]) = FLOAT4(a[(by * BM + load_a_row + 64) * k + bk + load_a_col]);

        FLOAT4(Bs[load_b_row][load_b_col]) = FLOAT4(b[(bk + load_b_row) * n + bx * BN + load_b_col]);
        FLOAT4(Bs[load_b_row + 8][load_b_col]) = FLOAT4(b[(bk + load_b_row + 8) * n + bx * BN + load_b_col]);

        __syncthreads();

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

The comments are fairly detailed, but let's emphasize a few key points:

- Loading the 128×16 A tile and 16×128 B tile is a straightforward flat thread mapping. Adjacent threads stride by 4 columns for `float4` vectorized loads, keeping adjacent threads on contiguous addresses for coalesced access.
  - Every 4 threads load one row of A; every 32 threads load one row of B.
- For C computation, I applied a 2D warp tiling: every 4 warps cover the upper and lower 64×128 halves. The initial design prioritized uniform, symmetric, contiguous row/column assignment. Each thread takes 8 contiguous rows of A and 8 contiguous columns of B for the accumulate-multiply inner loop.
- The C write-back uses contiguous columns, so `float4` stores work cleanly.

## 3. A-Transpose Tiling

The first kernel has an obvious optimization opportunity: the inner loop reads As from SMEM using scalar loads in a loop, creating significant LSU pressure. Frequently stalling for data during the compute loop creates many small pipeline bubbles. The immediate fix: transpose A before storing it in SMEM, enabling `float4` vectorized reads during computation and improving ILP.

Yes, this forces scalar writes during the load phase. But trading "2 `float4` loads unpacked into 8 scalar writes" in the outer loop for "16×8 scalar reads → 16×2 `float4` reads" in the inner compute loop is well worth it — significantly reducing LSU pressure and preventing ALU stalls waiting on L1 data.

The core change is simple — swap A's row and column indices:

```cpp
template <const int BM = 128, const int BN = 128, const int BK = 16, const int TM = 8, const int TN = 8>
__global__ void sgemm_at_tiling_kernel(float *a, float *b, float *c, int m, int n, int k) {
    // unchanged
    ...
    __shared__ float As_T[BK][BM]; // A transposed
    __shared__ float Bs[BK][BN];

    for (int bk = 0; bk < k; bk += BK) {
        // A: vectorized load from GMEM, scalar transpose-write into SMEM
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

#pragma unroll
        for (int i = 0; i < BK; i++) {
            float reg_a[TM], reg_b[TN];
            // float4 reads from transposed As
            FLOAT4(reg_a[0]) = FLOAT4(As_T[i][c_row]);
            FLOAT4(reg_a[4]) = FLOAT4(As_T[i][c_row + 4]);
            // rest unchanged
            ...
        }
        ...
    }
    ...
}
```

## 4. A-Transpose + Swizzle

The previous kernel improved inner-loop read efficiency by transposing As, but introduced a new problem: **shared memory bank conflicts**. The scalar writes into SMEM suffer from 4-way bank conflicts. Here's why:

```cpp
    int load_a_row = tid / 4;               // 0~63
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12...
    ...
    As_T[load_a_col + 0][load_a_row] = tmp_a0.x; // 4-way bank conflict
```

Why the conflict? Consider `As_T[16][128]`: each row has 128 elements, so the row index doesn't affect the bank ID (bank ID = (row × 128 + col) % 32). During the A load, threads are laid out flat for coalesced global access. When transpose-writing into SMEM, within a single warp the per-thread coordinates become (warp 0 as an example):

```yaml
col : 0, 0, 0, 0, 1, 1, 1, 1...
row : 0, 4, 8, 12, 0, 4, 8, 12...
```

Classic 4-way bank conflict — every four threads access different addresses in the same bank. We use a slightly involved swizzle technique to eliminate this.

We need a mapping f(row, col) → (row, new_col) that distributes new_col across all 32 banks. A simple `new_col = row ^ col` XOR swizzle (like in my matrix transpose article) won't work here, because neither col nor row alone traverses all 32 bank possibilities. XORing row bits into col directly doesn't achieve our goal.

Let's derive a proper swizzle formula. First, remember the physical constraint: Bank ID is determined by the **lower 5 bits** of the address (since % 32).

![swizzling](../static/swizzle.svg)

Observing carefully within a warp (warp 0 as an example):

- Row (i.e., `load_a_col` in the code) takes values 0, 4, 8, 12, ... — the variable bits are at positions 2 and 3 (`00000, 00100, 01000, 01100...`).
- Col (i.e., `load_a_row` in the code) ranges 0–7 — the variable bits are at positions 0–2 (`00000` through `00111`).

Naturally, since conflicts arise from identical col values while row values differ, we can left-shift row by 1 to produce 0, 8, 16, 24 — pushing its variable bits to positions 3 and 4. XOR this with col, and something magical happens: row fills the upper 2 bits (3, 4), col fills the lower 3 bits (0–2). Their interleaved XOR perfectly covers all 32 permutations of the lower 5 bits!

Other warps follow the same pattern. For instance, warp 1's col values are `01xxx` — XORing with the 4 permutations of bits 4–5 still produces 32 distinct values, guaranteed by XOR's bijective property. Within a warp, row changes are uniform (e.g., when row becomes 1, 5, 9, 13, binary `xx01`, left-shifted by 1 the variable bits remain at 4–5), so the same logic holds.

This gives us a conflict-free swizzle for As_T writes: **`new_col = col ^ (row << 1)`**

But this isn't enough. While the scrambled write perfectly distributes banks, the inner loop still needs `float4` vectorized reads. A `float4` load requires 4 contiguous floats that are physically adjacent and 16-byte aligned. Crucially, in `FLOAT4(reg_a[0]) = FLOAT4(As_T[i][SWIZZLE_A(i, c_row)])`, the outer dimension `i` (row) is **constant** across all 32 threads within a single iteration!

The fix: the lower 2 bits of the col index determine a 4-float segment. If we zero out row's lower 2 bits, then `00 XOR (col's bit 0–1) = col's bit 0–1` — preserving the relative order of `col+0, col+1, col+2, col+3`. This is the `float4`-compatible swizzle: **`new_col = col ^ ((row >> 2) << 3)`**.

This ensures row's perturbation bits stay at positions 4–5 (`>> 2` zeros out bits 0–1; when `i` iterates 0–3, `(i >> 2) << 3` is 0; when `i` iterates 4–7, it's 8; and so on).

Final core changes:

```cpp
#define SWIZZLE_A(x, y) ((y) ^ ((x >> 2) << 3))

template <const int BM = 128, const int BN = 128, const int BK = 16, const int TM = 8, const int TN = 8>
__global__ void sgemm_at_bcf_swizzling_kernel(float *a, float *b, float *c, int m, int n, int k) {
    // unchanged
    ...
    for (int bk = 0; bk < k; bk += BK) {
        // A: transpose + swizzle write into SMEM
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

#pragma unroll
        for (int i = 0; i < BK; i++) {
            // swizzled reads
            FLOAT4(reg_a[0]) = FLOAT4(As_T[i][SWIZZLE_A(i, c_row)]);
            FLOAT4(reg_a[4]) = FLOAT4(As_T[i][SWIZZLE_A(i, c_row + 4)]);

            ...
        }
        __syncthreads();
    }
}
```

A comprehensive treatment of swizzle mechanisms deserves its own article. But the key takeaway here: the specific swizzle formula must be **derived** from the data type and memory layout (CUTLASS's swizzle templates only work for specific data+layout combinations). Once you understand XOR's properties, deriving XOR swizzle formulas isn't hard — and you'll wield swizzle templates with much greater confidence.

## 5. A-Transpose + Swizzle + Fully Coalesced Global Memory R/W

The previous kernel is already quite strong for a single-buffer, non-pipelined implementation. But there's a blemish. NCU profiling reveals:

- Uncoalesced global memory write access, with warnings like:

```yaml
1.
The memory access pattern for global stores to L1TEX might not be optimal. On average, only 16.0 of the 32 bytes transmitted per sector are utilized by each thread. This could possibly be caused by a stride between threads. Check the  Source Counters section for uncoalesced global stores.

2.
This kernel has uncoalesced global accesses resulting in a total of 2097152 excessive sectors (2% of the total 138412032 sectors). Check the L2 Theoretical Sectors Global Excessive table for the primary source locations. The  CUDA Programming Guide has additional information on reducing uncoalesced device memory accesses.
```

Why does NCU flag this? Each thread computes 8 rows × 8 contiguous columns. When writing back to C, this requires two `float4` stores per row. Looking at adjacent threads within a warp: they're writing with 4-float gaps between them. Since L1/L2 memory transactions are 128 bytes each, our two-pass writes issue two identical transaction requests — the first writes 64 bytes, the second writes the other 64 bytes. Significant waste.

Both warnings reflect the same issue at L1 and L2 levels. The fix: redesign each thread's 8-column assignment. Split the 8 columns into two groups of 4 contiguous columns (still `float4`-compatible), and map adjacent threads to contiguous `float4` addresses. For example: T0 reads columns 0–3, T1 reads 4–7, ... After one pass, T0 reads 64–67, T1 reads 68–71, ... This ensures adjacent threads write to contiguous addresses.

![col_shuffle](../static/col_shuffle.svg)

Core changes — remove the 2D warp tiling, have each warp cover 16 rows of C, and map thread IDs directly to the desired c_col:

```cpp
    // warp tiling
    // Thread's row offset within the warp: 0 or 8
    int t_row_in_warp = (lane_id / 16) * 8;

    // Each warp covers 16 rows; every 16 threads handle 8 rows × 128 columns.
    // Each thread loads 8 rows × 8 columns, but the 8 columns are split into
    // two float4s spanning 64 columns apart. E.g., T0 handles cols 0-3 and 64-67.
    // This way, every 8 adjacent threads write 32 contiguous floats = 128 bytes = perfect coalescing.
    int c_row = warp_id * 16 + t_row_in_warp;
    int c_col_base = (lane_id % 16) * 4;
    int c_col_0 = c_col_base;      // 0~3
    int c_col_1 = c_col_base + 64; // 64~67

    ...

    FLOAT4(reg_b[0]) = FLOAT4(Bs[i][c_col_0]); // read 0~3
    FLOAT4(reg_b[4]) = FLOAT4(Bs[i][c_col_1]); // read 64~67
```

## 6. A-Transpose + Swizzle + Fully Coalesced R/W + Double SMEM Buffer

The previous kernel essentially hit the single-buffer ceiling: perfect coalesced access (GMEM and SMEM), strong compute-to-memory ratio. To push further, we need a classic technique: **copy-compute overlap** — the bread and butter of HPC. After the LSU issues memory transactions to load data from GMEM to registers (through L2 → L1 → register), the ALU sits idle. Even with multiple blocks/warps available for scheduling, overlapping loads and computes within a single warp is more efficient — further improving ILP and hiding memory latency. We implement this via traditional double buffering, without `cp.async` or other modern features.

**Pipeline algorithm:**

1. **Prologue:** Pre-load the first A/B tile into `smem_buffer[0]`. Sync with `__syncthreads()` to ensure SMEM writes complete.
2. **Main loop:**
   - Issue loads for the **next** A/B tile into registers.
   - Immediately compute using data in `smem_buffer[0]` (overlapping with the above loads).
   - After computation, write the register data into `smem_buffer[1]`. Sync with `__syncthreads()`.
     - Compared to the single-buffer kernel, the loop body has **one fewer sync** (classic space-time tradeoff).
   - Swap buffer pointers; begin next iteration.
3. **Epilogue:** After the loop, compute the last loaded tile. Write the accumulated `sum[8][8]` back to C in GMEM.

Here's the final, complete kernel:

```cpp

template <const int BM = 128, const int BN = 128, const int BK = 16, const int TM = 8, const int TN = 8>
__global__ void sgemm_at_bcf_swizzling_dbf_rw_kernel(float *a, float *b, float *c, int m, int n, int k) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x; // 0~255; 8 warps
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    int load_a_row = tid / 4;               // 0~63
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12...
    int load_b_row = tid / WARP_SIZE;       // 0~8
    int load_b_col = (tid % WARP_SIZE) * 4; // 0,4,8,12,16,20,24,28...

    // C compute/store mapping — same as before
    int t_row_in_warp = (lane_id / 16) * 8;
    int c_row = warp_id * 16 + t_row_in_warp;
    int c_col_base = (lane_id % 16) * 4;
    int c_col_0 = c_col_base; // 0~3
    // int c_col_1 = c_col_base + 64; // 64~67 — eliminated to reduce register pressure

    // double buffer
    __shared__ float As_T[2][BK][BM];
    __shared__ float Bs[2][BK][BN];

    float sum[TM][TN] = {0.f};

    // Flat pointers for GMEM loads, easy to advance in the pipeline
    float *a_ptr = a + (by * BM + load_a_row) * k + load_a_col;
    // float *a_ptr_64 = a + (by * BM + load_a_row + 64) * k + load_a_col;
    float *b_ptr = b + load_b_row * n + bx * BN + load_b_col;
    // float *b_ptr_8 = b + (load_b_row + 8) * n + bx * BN + load_b_col;

    // Prologue: load first tile (costs 16 extra registers;
    // commented-out vars above keep register count below 128 to preserve occupancy)
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

    int write_idx = 1;
    int read_idx = 0;
    // Main loop
    for (int bk = BK; bk < k; bk += BK) {
        a_ptr += BK;
        b_ptr += BK * n;

        // Issue loads for the next tile — asynchronous; computation can begin immediately
        tmp_a0 = FLOAT4(a_ptr[0]);
        tmp_a1 = FLOAT4(a_ptr[64 * k]);
        tmp_b0 = FLOAT4(b_ptr[0]);
        tmp_b1 = FLOAT4(b_ptr[8 * n]);

        // Compute logic — identical to before
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

        // Write the async-loaded register data into the other SMEM buffer
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

        __syncthreads();
        write_idx ^= 1;
        read_idx ^= 1;
    }
    // Epilogue: compute the last loaded tile
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
    // Pipeline complete — write back C
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        FLOAT4(c[(by * BM + c_row + i) * n + bx * BN + c_col_0]) = FLOAT4(sum[i][0]);
        FLOAT4(c[(by * BM + c_row + i) * n + bx * BN + c_col_0 + 64]) = FLOAT4(sum[i][4]);
    }
}
```

Careful readers may notice that the final code comments out several intermediate variables (`c_col_1`, `a_ptr_64`), computing offsets inline instead (e.g., `a_ptr[64 * k]`). Double buffering doubles the read/write register footprint. Without careful budgeting, it's easy to exceed the per-thread register threshold (my initial code exceeded 128 registers per thread, halving active warps and tanking performance). By eliminating these unnecessary intermediates, I pushed register usage back under 128, preserving occupancy.

## 7. Benchmark Results and Analysis

Choosing M=N=K=4096 was deliberate. First, as a perfect power of 2, it lets cuBLAS run on its fast path with zero boundary-check overhead — showcasing its full strength. Second, 4096 is the standard hidden dimension for today's mainstream 7B/8B large language models, extremely common during the prefill stage. Going head-to-head on "cuBLAS's home turf" and "modern AI's most critical compute scenario" makes the benchmark all the more meaningful.

Enough theory — claims about theoretical performance are worthless without numbers. Here are the benchmark results and NCU profile report. Test device: RTX 5060 Laptop (absolute timings may fluctuate due to dynamic clocking and background processes).

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
####################################################################################################
sgemm_cublas_tf32              mean time:  8.798057 ms, speedup: 1.70
```

From 18.76 ms to 14.19 ms — a purely hand-crafted optimization journey, going head-to-head with cuBLAS and coming out on top! Throughout this process, we used precise resource allocation, intricate and decoupled coordinate mappings for data movement and computation (each different for A, B, and C), and double buffer pipelining to surpass cuBLAS. Master these techniques, and you'll have the confidence to tackle virtually any matrix-related kernel implementation.

### NCU Report

![ncu report](../static/sgemm_ncu.png)

The report shows that NCU is completely satisfied with `sgemm_at_bcf_swizzling_dbf_rw`: **`estimated speedup = 0%`** — meaning NCU considers the kernel at its theoretical performance ceiling. Meanwhile, the cuBLAS kernel is the one with "room for improvement." :)

## 8. Discussion

The NCU report reveals several interesting observations:

- The first two swizzle kernels still report `shared store bank conflict`. While I repeatedly hand-computed coordinates and verified there shouldn't be any intra-warp conflicts, controlled experiments showed the warning disappears when Bs writes are commented out. (**Update:** The root cause is inter-warp bank conflicts — different warps accessing different addresses in the same bank. This makes sense: all 256 threads are simultaneously writing 8×128 data, and an SM has 4 sub-core schedulers, so there's a real probability of cross-warp bank collisions. See [NVIDIA GTC talk](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41723/) for details.)

- Interestingly, this conflict warning **disappears** in the double buffer kernel (despite identical SMEM write logic). Why? Digging into `Memory Workload Analysis`, the phantom bank conflict is still there — it's just no longer flagged as a key performance bottleneck. The takeaway: when a hardware-level limitation can't be eliminated, **architectural design can mask it**. The double buffer pipeline hides this ineliminable hardware conflict latency beneath the massive compute stream, to the point where even NCU no longer considers it a bottleneck.

- The report also reveals that cuBLAS dispatches a CUTLASS kernel: `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_align1>(T1::Params)`. CUTLASS SIMT naming convention: `[M]x[N]_[K]x[Stages]`.
  - cuBLAS uses 128×64 tiling (M=128, N=64), K-stride of 8, and an aggressive **5-stage pipeline** for latency hiding.
  - Block size is 128 (corresponding to smaller data loads); grid is 512×4 (likely using grid swizzling to reorder block-to-GMEM mapping for L2 cache locality).
  - Opening cuBLAS's `Memory Workload Analysis` reveals massive `shared store from global load bank conflict` — meaning cuBLAS uses `cp.async` for asynchronous data loading and **also tolerates bank conflicts**, because the hardware async copy path (`cp.async`) has different transaction packing and bandwidth characteristics than traditional LDG + SMEM stores. On one hand, it's unavoidable; on the other, it demonstrates that with sufficient pipeline depth, this level of conflict latency is fully hidden — further validating our double buffering approach.

- **Tensor Core supremacy.** Our hand-crafted SIMT kernel runs at 14.19 ms, while cuBLAS TF32 achieves 8.79 ms. This shows that Tensor Cores nearly **double** the compute ceiling.

## Conclusion

In an era where Tensor Cores dominate large-model compute, pushing the SIMT programming model to its limits isn't just about shaving one or two milliseconds off cuBLAS. It's about maintaining a deep mental model of the underlying hardware. With that understanding and mastery, wielding new architectures and features becomes second nature.

Corrections and feedback are welcome. Feel free to star [Vitamin-CUDA](https://github.com/WingEdge777/Vitamin-CUDA) and join the discussion.

That's all.
