# [CUDA in Practice] SGEMM TF32 — Beating cuBLAS with Tensor Cores, cp.async, ldmatrix & mma

## 0. Preface — The Era of Vectorized Compute

> **Heads up:** This is an intense, diagram-heavy deep dive. It covers hardcore swizzle derivations (if you still don't understand XOR swizzle after reading this, come find me), layout-to-address coordinate mapping, and detailed PTX instruction analysis. Best read on a desktop for the full experience.
>
> **Target audience:** Readers with a solid CUDA programming foundation, familiar with GEMM optimization, and interested in advanced Tensor Core / inline PTX performance tuning.
>
> All kernel source code is available on GitHub. Feel free to check out my hand-crafted kernel series, the [**vitamin-cuda**](https://github.com/WingEdge777/Vitamin-CUDA) project.
>
> Starting to feel like this is becoming a "Beating cuBLAS" anthology...
>
> In today's world where Tensor Cores are everywhere, if you still don't know how to use them for GEMM, you might be falling behind.

This article takes an M=N=K=4096 (M×K×N — the mid-size regime where cuBLAS excels) TF32 GEMM as the running example. On an RTX 5060 Laptop GPU, I use `cp.async`, `ldmatrix`, and `mma` PTX instructions combined with Tensor Core TF32 acceleration to **beat NVIDIA's own cuBLAS at equal precision**. This post is a retrospective of that battle — and it was a real battle, driven by iterative NCU profiling at every step. The details go down to `cp.async` semantics, warp-level `ldmatrix`/`mma` PTX usage, a rather hardcore swizzle derivation (even though it didn't all get used in the end), and layout analysis.

The article presents **five kernel implementations**, progressing from a basic `cp.async` + dual `ldmatrix` + `mma` baseline, through swizzle-based bank conflict elimination, reverse-engineering cuBLAS strategies for better SMEM access, grid swizzling for L2 reuse, and double buffering for latency hiding — ultimately surpassing cuBLAS. Throughout, I explain instruction requirements, usage patterns, and the craft of swizzle design, hoping to help readers deeply understand Tensor Core usage and optimization techniques.

Unlike my previous SGEMM kernel series — where I already knew the optimization path and wrote the code accordingly (I'm fairly experienced with GEMM optimization) — this time I was not familiar with Tensor Cores. The entire journey was **NCU-profile-driven**. I started by profiling the cuBLAS kernel, reverse-engineered its strategy from the kernel name and shared memory statistics, and eventually beat it. (I have reasonable grounds to suspect cuBLAS hasn't been updated in a while — or at least hasn't been optimized for consumer GPUs — because I matched its performance before even deploying all my optimization tricks.)

Here is the kernel roadmap (the first entry is the cuBLAS baseline):

- `sgemm_cublas` — cuBLAS TF32 baseline
- `sgemm_tf32_bt` — vectorized A/B loads, B transposed into SMEM, dual `ldmatrix` + `mma`
- `sgemm_tf32_bt_swizzle` — + As zero bank conflicts via swizzle
- `sgemm_tf32_bt_swizzle_dbf` — + grid swizzling, 97–102% of cuBLAS
- `sgemm_tf32_swizzle_bcf` — `cp.async` for both A/B, swizzle, As/Bs conflict-free, grid swizzling
- `sgemm_tf32_swizzle_bcf_dbf` — + double buffer, **outperforms cuBLAS**

## 1. PTX Instructions & NCU Profiling the cuBLAS Kernel

When I set out to write the SGEMM TF32 kernel, I first searched for blog posts about `ldmatrix` and `mma` to learn from. After browsing around, I found that most content covers half-precision GEMM — which makes sense, since Tensor Cores were originally designed for FP16. But does nobody care about TF32? Left with no choice, I dove into NVIDIA's PTX documentation and profiled the cuBLAS kernel as a reference.

### PTX Instructions

The PTX (Parallel Thread Execution) official documentation lives at: <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>. PTX is NVIDIA's virtual assembly ISA. The key instructions used in this article are `cp.async`, `ldmatrix`, and `mma`. `cp.async` is an asynchronous copy instruction; `ldmatrix` and `mma` are warp-level cooperative load/compute instructions.

Before diving into the instructions, let's briefly cover the syntax for embedding PTX assembly in C++ (CUDA) code. It uses GCC extended inline assembly syntax:

```cpp
asm volatile(
    "assembly instruction template;"
    : comma-separated output operands   /* optional */
    : comma-separated input operands    /* optional */
    : clobber descriptors               /* optional */
);
```

For example, the `cp.async` instruction we'll be using:

```cpp
asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(dst_smem_32b), "l"(src_global_ptr))
```

Key keywords:

- `asm`: Tells the compiler that an inline assembly block follows.
- `volatile`: Prevents the compiler from reordering or eliminating this block (e.g., it won't be dead-code-eliminated even if the output appears unused).

Punctuation and placeholders:

- `%0`, `%1`: Placeholders in the assembly string. The compiler maps C++ variables to these in order, starting from 0, based on the output and input operand lists.
- `\n`: Purely cosmetic — ensures clean line breaks in the generated assembly listing.
- `:` and `::`: Colons separate the assembly template, outputs, inputs, and clobbers. If an instruction has only inputs and no outputs, use `::` to skip the output section.
- **Clobber descriptors** explicitly tell the compiler which physical registers, memory, or flags the instruction may modify, forcing the compiler to invalidate cached values and reload them, preventing subtle logic bugs.

Constraint characters — when binding C++ variables to assembly operands, we use constraint strings to guide register allocation:

- `=` (modifier): "Write-only" — used for output operands. Without `=`, the default is "read-only" (input).
- `r` (Register): Allocate a 32-bit general-purpose integer register.
- `l` (Long): 64-bit register (in CUDA, typically used for 64-bit global memory pointers like `src_global_ptr`).
- `f` (Float): 32-bit floating-point register.
- `"=r"`: Output to a 32-bit GPR. `"r"`: Read from a 32-bit GPR. `"=f"` / `"f"`: Same idea for FP registers.

Now let's look at the specific instructions we'll use. I'll only cover the exact forms relevant to our kernels and skip exhaustive operand lists to keep things concise — refer to the official docs for the full picture.

```cpp
cp.async.cg.shared.global.L2::128B [%0], [%1], 16; // cg: Cache at Global level — bypasses L1, copies 16 bytes
cp.async.commit_group;  // commit the async copy group
cp.async.wait_group 0;  // 0 means wait for ALL groups to complete (no groups allowed in-flight)
```

- **`cp.async`**: Asynchronous copy instruction. Bypasses L1 and registers, streaming data from GMEM through L2 directly into SMEM. Saves register pressure and its asynchronous nature makes it ideal as a building block for software pipelines.

```cpp
mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32        d, a, b, c;
```

- **`mma`**: Before explaining `ldmatrix`, we need to understand `mma`, since `ldmatrix` exists to serve `mma`.
  - The A sub-matrix input is row-major; the B sub-matrix is column-major.
  - Per the official docs, TF32 precision (added later) only supports the shape `m16n8k8`.
  - Once the shape is fixed, the fragment shapes and index mappings for A, B, and C are fully determined.

  In short, `mma` is somewhat rigid (in my opinion).

```cpp
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4]; // cooperatively load 4 8x8 matrices; each thread ends up holding 4 values
ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2]; // cooperatively load 2 8x8 matrices; each thread ends up holding 2 values
```

- **`ldmatrix`**: A warp cooperatively loads a small matrix tile from SMEM into registers distributed across all threads. The collective set of registers held by all threads forms a **fragment** (rigid +1).
  - Note: `x2` and `x4` mean 16 and 32 threads provide addresses, respectively. Each address-providing thread **always reads 16 bytes**. After the read, data is redistributed across threads to form the fragment. Understanding this is essential to resolving bank conflicts.
  - Since TF32 `mma` only supports `m16n8k8`, to load a 16×8 A tile and an 8×8 B tile, we must use the `.m8n8` shape with `.b16` type to move 32-bit data.
  - Also due to TF32, we cannot use the `.trans` modifier for transposition (another gotcha — manual transposition and register-level shuffle attempts cost me a lot of time).

### cuBLAS Kernel NCU Report

![](../static/cublas_tf32_ncu_0.png)

![](../static/cublas_tf32_ncu_1.png)

Honestly, when I first saw the cuBLAS NCU report, I was intimidated. Non-trivial compute/memory throughput, a flawless shared memory statistics table, use of `cp.async` and `ldmatrix` with shared loads, and **zero bank conflicts**. On top of that: fully coalesced global memory access, a 4-stage pipeline, and registers pushed hard to 228 per thread. It looked like there was no room left to optimize. How could I possibly match this? Initially, I figured reaching 95% of cuBLAS would be a good outcome.

Looking at the kernel name — it dispatches a CUTLASS kernel: `void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_64x256_16x4_nn_align4>(T1::Params)`. So: `mma m16n8k8`, 64×256 tiling (M=64, N=256), K-dimension tile of 16, 4-stage pipeline.

OK — initial recon on the opponent complete. Time to write our own kernel.

## 2. sgemm_tf32_bt

For this first kernel, the plan was simple: get `cp.async` working, fire up `ldmatrix` + `mma`, and call it a day. Bank conflicts, coalesced access — worry about those later. While I could roughly guess CUTLASS's implementation, I didn't want to copy it. Copying won't beat the master, and the reverse-engineered strategy might not even be complete. The 4-stage pipeline was also out — I'd need to resize SMEM and manage 4-stage scheduling. Pass.

![](../static/tiling.svg)

That said, it wasn't entirely rough. I followed the same approach as my previous SGEMM kernel:

- 128×128 tiling (from C's perspective), K-dimension stride of 16, block size of 256 threads (tile size selection involves many factors — see my previous article for details).
- A 2×4 warp tiling scheme, splitting the C tile into upper and lower 64×128 halves. Each warp handles a 64×32 block of C. Mapped to `mma`'s 16×8×8: 64/16 = 4, 32/8 = 4, giving 4×4 rounds (K-dimension accumulation counted separately). Perfectly square and symmetric — just how I like it.

The overall flow is straightforward:

- Load As from global memory via `cp.async` (bypassing L1, direct to SMEM). Load Bs via LDG + manual transpose into SMEM.
- Use `ldmatrix` to load As and Bs into registers. Compute with `mma`.
- Obtain the C fragment and write it back.

One thing worth mentioning upfront: the C fragment layout (i.e., which values each thread within a warp holds after `mma`) must be understood before you can map results back to global coordinates. See: <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688>

Code:

```cpp
// a block calculate c[128][128]
template <const int BM = 128, const int BN = 128, const int BK = 16>
__global__ __launch_bounds__(256, 2) void sgemm_tf32_bt_kernel(float *a, float *b, float *c, int m, int n, int k) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x; // 0~255
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    int load_a_row = tid / 4;               // 0~63
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12
    int load_b_row = tid / WARP_SIZE;       // 0~7  (K dimension)
    int load_b_col = (tid % WARP_SIZE) * 4; // 0~124 (N dimension)

    // A stays row-major; B is transposed to column-major
    __shared__ float As[BM][BK];
    __shared__ float Bs[BN][BK];

    // 2x4 warp tiling
    // One row of warps covers the upper/lower 64x128; each warp handles 64x32 of C
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // Total registers: M-dim 4 tiles * N-dim 4 tiles * 4 regs per tile = 64
    float sum[4][4][4] = {0.f};

    for (int bk = 0; bk < k; bk += BK) {

        // 1. Load A via cp.async (16-byte aligned)
        uint32_t smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row][load_a_col]));
        uint32_t smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row + 64][load_a_col]));

        float *global_a0 = &a[(by * BM + load_a_row) * k + bk + load_a_col];
        float *global_a1 = &a[(by * BM + load_a_row + 64) * k + bk + load_a_col];

        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);
        CP_ASYNC_COMMIT_GROUP();

        // 2. Load B and manually transpose into SMEM
        float4 tmp_b0 = FLOAT4(b[(bk + load_b_row) * n + bx * BN + load_b_col]);
        float4 tmp_b1 = FLOAT4(b[(bk + load_b_row + 8) * n + bx * BN + load_b_col]);

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

        // 3. Tensor Core compute (2 K-steps, each consuming 8 K elements)
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 8;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // Issue 4 ldmatrix to load A tiles (4 * 16 = 64 rows)
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
                int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
                int a_col = k_offset + (lane_id / 16) * 4;
                uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&As[a_row][a_col]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

            // Issue 4 ldmatrix to load B tiles (4 * 8 = 32 columns)
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                int b_row = warp_id_n * 32 + n_idx * 8 + (lane_id % 8);
                int b_col = k_offset + ((lane_id / 8) % 2) * 4;
                uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[b_row][b_col]));
                LDMATRIX_X2(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
            }

            // MMA core: 4x4 grid of m16n8k8
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

    // Map Tensor Core m16n8k8 C fragment back to global coordinates
    // c fragments layout:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688 1688.tf32
    int t_row = lane_id / 4;       // 0~7
    int t_col = (lane_id % 4) * 2; // 0, 2, 4, 6

#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int c_base_row = by * BM + warp_id_m * 64 + m_idx * 16;
            int c_base_col = bx * BN + warp_id_n * 32 + n_idx * 8;

            FLOAT2(c[(c_base_row + t_row) * n + c_base_col + t_col]) = FLOAT2(sum[m_idx][n_idx][0]);
            FLOAT2(c[(c_base_row + t_row + 8) * n + c_base_col + t_col]) = FLOAT2(sum[m_idx][n_idx][2]);
        }
    }
}
```

Benchmark results:

```yaml
n: 4096, m: 4096, k: 4096
torch                                    mean time: 15.146454 ms, 9.07 tflops
sgemm_cublas_tf32                        mean time: 8.535476 ms, speedup: 1.77, tflops: 16.10
sgemm_tf32_bt                            mean time: 15.925327 ms, speedup: 0.95, tflops: 8.63
```

Ouch — even slower than PyTorch's FP32 matmul. Let's profile with NCU:

![](../static/bt.png)

The glaring bank conflict ratio (average 32.6-way — I'm not even sure how the hardware counted that) was expected. Time to optimize SMEM access.

## 3. sgemm_tf32_bt_swizzle

### As Shared Memory Optimization

First, let's understand how data is loaded into As and how `ldmatrix` accesses it. The 128×16 rows of data are loaded via `cp.async` — think of it as a threaded `float4` vectorized copy (except we bypass L1 and registers). The important part is understanding how `ldmatrix` reads As. After reading the official docs, here's my summary.

A warp executing `ldmatrix` has two phases:

1. Designated threads read 16 bytes each from their respective addresses (`x1`/`x2`/`x4` use the first 8/16/32 threads respectively).
2. The fetched data is distributed among all 32 threads.

Our warp tiling works out nicely: we can use `ldmatrix x4`, where all 32 threads read a 16×8 tile.

```cpp
int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
int a_col = k_offset + (lane_id / 16) * 4;
```

As is a 128×16 float array. To load a 16×8 matrix, we provide 16 rows × 2 columns = 32 addresses (each pointing to the start of 16 contiguous bytes; the two columns are 4 floats apart — effectively 16 rows of 8 contiguous floats). The per-thread addresses (warp 0, m_idx=0, k_offset=0):

```yaml
row:0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
col:0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4
```

A 32-thread access of 512 bytes is a wide memory transaction. The GPU splits it into 4 phases, 8 threads per phase. Looking at just the first 8 threads:

```yaml
row:0, 1, 2, 3, 4, 5, 6, 7
col:0, 0, 0, 0, 0, 0, 0, 0
```

Clearly, accessing the same column on alternating rows hits the same banks — 4-way bank conflict.

Time to derive our XOR swizzle formula. We need a mapping f(row, col) → (row, new_col) that eliminates these conflicts. Observing closely:

- Row values in binary: `00000`–`01111` (`0xxxx`); the first 8 threads are `00000`–`00111`.
- Column values are identical: `00000` (looking at the first 8 threads only).

![](../static/swizzle_a.svg)

Bank conflicts arise from different addresses hitting the same bank ID. Specifically here, rows with the same column value map to conflicting banks (somewhat harder to reason about since the SMEM width spans 16 banks). Since rows differ, we can use row bits to perturb the column.

First, we know that every two rows span 32 banks, so row pairs form a group — meaning row bit 0 is irrelevant. Zero it out: `(row >> 1) << 1`. Binding row pairs (both rows get the same column offset) frees space to stagger against other rows.

Second, `cp.async` requires 16-byte (`float4`) alignment, so we cannot perturb col bits 0–1. Can we just use `col ^ ((row >> 1) << 2)`? No — our array is 128×16; valid `float4`-aligned column addresses are only 0, 4, 8, 12. The perturbation must not push column bits beyond bit 4 (that would go out of bounds). So we keep only 2 effective bits from `row >> 1`: `((row >> 1) & 0x3) << 2`.

This pushes row bits 1–2 into column bits 2–3, yielding values {0, 4, 8, 12}. Since the original column is uniform, XOR cycles through all four bit 2–3 permutations.

Our bank-conflict-free swizzle for As: **`new_col = col ^ (((row >> 1) & 0x3) << 2)`**

Different `m_idx`, `k_offset` values, other phase groups, and other warps all follow the same pattern — the principle is identical; only certain bits flip uniformly across the group.

You might say this feels like "just fiddling with bits." You'd be right — I carefully crafted this XOR swizzle by exploiting XOR's properties, and it took a while. The meaning? Under hardware alignment constraints, perturb column bits using row bits so that addresses spread across banks {0, 4, 8, 12}. That's the essence of XOR swizzle design.

Core changes:

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

### Bs Shared Memory Optimization

Now let's look at the global-to-Bs write path. Due to the transpose, the array is again 128×16 (seeing this layout triggers PTSD at this point), and writes are scalar:

```cpp
int load_b_row = tid / WARP_SIZE;       // 0~7  (K dimension)
int load_b_col = (tid % WARP_SIZE) * 4; // 0~124 (N dimension)
```

Per-thread write addresses (warp 0):

```yaml
row:0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124
col:0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
```

Obviously catastrophic — 32-way bank conflict.

What about the Bs read side? `ldmatrix x2` loads an 8×8 matrix using 16 addresses (8 rows × 2 columns, similar to As):

```cpp
int b_row = warp_id_n * 32 + n_idx * 8 + (lane_id % 8);
int b_col = k_offset + ((lane_id / 8) % 2) * 4;
```

Per-thread addresses (warp 0, n_idx=0, k_offset=0; only the first 16 threads, since the other 16 don't directly read SMEM):

```yaml
row:0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7
col:0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4
```

Again a wide memory transaction, split into 2 phases of 8 threads. Same story: 4-way bank conflict.

At this point I was nearly ready to give up — it seemed impossible to eliminate conflicts from both the LDG write and the `ldmatrix` read simultaneously. But there was no choice except to press on, starting with the easier side.

The `ldmatrix` read is simpler to fix: `ldmatrix x2` is essentially the first-16-thread subset of `ldmatrix x4` on As. So the As swizzle applies directly, and the Bs read conflict is eliminated.

We now have `new_col = col ^ (((row >> 1) & 0x3) << 2)`. Not bad — one step took care of the Bs read side.

But does this swizzle work for the LDG write? Let's look. The row values in binary during LDG writes:

```yaml
00000, 00100, 01000, 01100, 10000, 10100, 11000, 11100...
```

We're extracting row bits 1–2 and shifting them to bits 2–3. For row values 0, 4, 8, 12, the XOR offset has only two distinct states: `0000` and `1000`. All 32 threads land on just two banks — a brutal 16-way conflict.

![](../static/swizzle_b.svg)

So what do we do? Looking more carefully, row is always a multiple of 4; its effective variable bits are at positions 2–4. If we use bits 2–4 directly: `((row >> 2) & 0x7) << 2`. Still no good — this would go out of array bounds (as discussed for As, the perturbation range must stay within bits 2–3). Constraining: `((row >> 2) & 0x3) << 2`.

Looks promising, but there's a problem: `row >> 2` bundles four rows together (all four get the same column offset), which creates a 2-way conflict for `ldmatrix x2`.

To use row's effective bits 2–4 **without** breaking `ldmatrix`'s two-row pairing, we need a trick. Here comes the black magic:

- Extract row bits 1–2 and bits 3–4, XOR them together: `((row >> 1) & 0x3) ^ ((row >> 3) & 0x3)` = `((row >> 1) ^ (row >> 3)) & 0x3`
- Shift the result to bits 2–3: `(((row >> 1) ^ (row >> 3)) & 0x3) << 2`

This reduces the LDG write conflict to 8-way (fully cycling through the four permutations of bits 2–3).

`ldmatrix x2` remains conflict-free: `row >> 1` preserves the two-row pairing, and row's upper bits 3–4 are constant within each 8-thread phase, so XOR may flip them but does so uniformly for all 8 threads — the pairing is unaffected.

Other `n_idx`, `k_offset`, and warp values follow the same swizzle, for the same reason. At this point the derivation is complete — I can't push it further. To my understanding, 8-way is the optimization limit for the LDG write: the array is 16 floats wide with a `float4` alignment constraint, leaving only 4 possible offsets, so 8-way is the floor. (I could be wrong — corrections welcome. This is as far as my analysis could reach.)

You might still ask: what's the mathematical/physical meaning of all this bit manipulation? My interpretation: I'm exploiting XOR's mathematical properties to extract effective variable bits from the row index and use them to perturb column bits, spreading accesses across banks. XOR's **bijective** property guarantees that writes never collide with each other; its **self-inverse** property (`row ^ col ^ col = row`) ensures read-write symmetry, making the scheme easy to use. I'd encourage readers to demystify XOR swizzle — understand XOR's properties, and you can derive your own. The key principle is **using the row's effective variable bits to perturb column bits**. Even CUTLASS's templated swizzle abstractions only work for specific data types and layouts.

Our final Bs swizzle: **`new_col = col ^ ((((row >> 1) ^ (row >> 3)) & 0x3) << 2)`**

Core changes:

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

Benchmark:

```yaml
n: 4096, m: 4096, k: 4096
torch                                    mean time: 15.146454 ms, 9.07 tflops
sgemm_cublas_tf32                        mean time: 8.535476 ms, speedup: 1.77, tflops: 16.10
sgemm_tf32_bt                            mean time: 15.925327 ms, speedup: 0.95, tflops: 8.63
sgemm_tf32_bt_swizzle                    mean time: 9.786451 ms, speedup: 1.55, tflops: 14.04
```

Solid improvement. Just 1–2 ms shy of cuBLAS.

## 4. sgemm_tf32_bt_swizzle_dbf

At this point we have conflict-free As reads/writes, and Bs conflicts are significantly reduced. I couldn't think of a way to fully solve the Bs conflicts, and I'd heard rumors that `ldmatrix` uses a different hardware path than LDG. Clinging to that hope — maybe double buffering can mask the remaining conflict latency (cuBLAS SIMT didn't solve all its conflicts either) — I went ahead and added the double buffer pipeline as a Hail Mary. The pipeline structure follows my previous article; I'll skip the detailed code. Straight to the benchmark:

```yaml
n: 4096, m: 4096, k: 4096
torch                                    mean time: 15.146454 ms, 9.07 tflops
sgemm_cublas_tf32                        mean time: 8.535476 ms, speedup: 1.77, tflops: 16.10
sgemm_tf32_bt_swizzle_dbf                mean time: 9.025055 ms, speedup: 1.68, tflops: 15.23
```

![](../static/cublas_l2.png)

![](../static/L2.png)

Closer, but still ~0.x ms short. Comparing NCU reports, I noticed cuBLAS had exceptionally high L2 cache hit rates (85%+) while mine was only 30%+. Could this be the gap?

### Grid Swizzling

Whether L2 matters — let's find out. Enter: **grid swizzle**. Grid swizzling reorders the sequence in which blocks access global memory tiles. By default, CUDA launches blocks in X-dimension (N) first, then Y-dimension (M).

![](../static/grid_swizzle.svg)

We manually remap each block's actual tile assignment, folding the traversal into a vertical strip of width 8. After processing 8 tiles along the X dimension, the traversal immediately drops to the next row. A full A-row / B-column tile is 128 × 4096 × 4 bytes = 2 MB. Without the width constraint, the GPU executes all 32 blocks in row 0. These 32 blocks share 1 A tile (2 MB) but each needs a different B tile.

That means computing row 0 requires reading 66 MB total — exceeding my L2 cache (32 MB)! After ~15 blocks, L2 is flushed by B tiles.

By limiting to 8 X-blocks before advancing to the next row, each batch of 8 blocks reads 1 A tile (2 MB) + 8 B tiles (16 MB) = 18 MB total, which fits entirely in L2. The next row's 8 blocks load a new A tile but fully reuse those 8 B tiles, avoiding redundant global memory reads — maximizing L2 hit rate.

Implementation:

```cpp
    // grid swizzling
    int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8;

    int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    int by = (linear_id / SWIZZLE_W) % gridDim.y;
```

Benchmark and NCU:

```yaml
n: 4096, m: 4096, k: 4096
torch                                    mean time: 15.146454 ms, 9.07 tflops
sgemm_cublas_tf32                        mean time: 8.535476 ms, speedup: 1.77, tflops: 16.10
sgemm_tf32_bt_swizzle_dbf                mean time: 8.723189 ms, speedup: 1.83, tflops: 15.76
```

![](../static/l2_opt.png)

It works. Latency improved further, and NCU shows our L2 hit rate jumped to 90%! Unfortunately, still a hair behind cuBLAS.

## 5. sgemm_tf32_swizzle_bcf

So close. SMEM swizzle, double buffer, grid swizzle — all deployed. What's left?

I stared at the profiles, thinking hard. Opened the cuBLAS NCU report side by side to find the biggest remaining gap.

![](../static/cublas_sh.png)
![](../static/sh.png)

cuBLAS: **0 bank conflicts**. Me: **150M+ conflicts**. That's the gap. There's no reason cuBLAS can do it and we can't. How does cuBLAS achieve it? Comparing the shared memory statistics tables, I noticed:

- cuBLAS's `ldmatrix` instruction count is 1/4 of mine, with significantly more regular shared loads.
- cuBLAS's `cp.async` count is 2× mine, with zero LDG instructions.

The implication: cuBLAS uses `cp.async` for **all** global memory reads, and some SMEM reads use regular shared loads instead of `ldmatrix` — clearly, Bs is read via standard shared load (the 1/4 `ldmatrix` count likely relates to cuBLAS's 64×256 tiling, which reuses SMEM reads and registers more aggressively).

Having reverse-engineered cuBLAS's Bs read strategy, let's adopt it. The allure of 0 conflicts is irresistible. We'll switch Bs from LDG to `cp.async`, and replace `ldmatrix` with regular shared loads.

This step isn't trivial — we need to understand the exact B fragment layout after `ldmatrix` to replicate it via direct SMEM reads, or `mma` won't work. From the official docs:

![](../static/b_fragment.png)

Let's carefully understand the B fragment register layout for `m16n8k8` TF32. In the diagram, b0 and b1 denote each thread's two 32-bit registers. After loading an 8×8 matrix, every 4 threads hold one column: T0 holds b[0][0] and b[4][0]; T1 holds b[1][0] and b[5][0]; ... T4 holds b[0][1] and b[4][1]; etc.

Here's my diagram for clarity:

![](../static/thread_b_reg.svg)

Not that complicated when you see it laid out (key insight: row/col refer to the logical indices of the B sub-matrix, independent of whether the memory layout is row-major or column-major). We can simply load data from GMEM into Bs as-is, then manually compute the coordinate mapping when reading.

Let's modify the single-buffer kernel. Core changes: remove the Bs transpose, change to a 16×128 layout (much more comfortable), use `cp.async` for GMEM → Bs, then read SMEM directly with coordinate mapping.

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
// Inner loop: read Bs
#pragma unroll
for (int n_idx = 0; n_idx < 4; ++n_idx) {
    int n_base = warp_id_n * 32 + n_idx * 8;

    // Every 4 threads share one column
    int b_col = n_base + (lane_id / 4);

    // K-dimension rows 0-3 and 4-7
    int b_row_0 = k_offset + (lane_id % 4);
    int b_row_1 = k_offset + (lane_id % 4) + 4;

    // Swizzled reads
    reg_b[n_idx][0] = __float_as_uint(Bs[b_row_0][SWIZZLE_B_F2(b_row_0, b_col)]);
    reg_b[n_idx][1] = __float_as_uint(Bs[b_row_1][SWIZZLE_B_F2(b_row_1, b_col)]);
}
```

Note that we've introduced a brand new swizzle macro `SWIZZLE_B_F2`. I'll spare the detailed bit-manipulation derivation here (same principle as before — try it yourself as an exercise). The two core design goals are:

- Satisfy `cp.async`'s 16-byte alignment requirement.
- Ensure that the 4 threads sharing a column are spread across 8 distinct banks.

Benchmark:

```yaml
n: 4096, m: 4096, k: 4096
torch                                    mean time: 15.146454 ms, 9.07 tflops
sgemm_cublas_tf32                        mean time: 8.535476 ms, speedup: 1.77, tflops: 16.10
sgemm_tf32_bt_swizzle_dbf                mean time: 8.723189 ms, speedup: 1.83, tflops: 15.76
sgemm_tf32_swizzle_bcf                   mean time: 8.650843 ms, speedup: 1.83, tflops: 15.89
```

Liftoff — the single-buffer version is even faster than the previous double-buffered one.

![](../static/sh_opt.png)

Bank conflicts: effectively zero (NCU shows 48,618 / 100,715,659 ≈ 0.04% — caused by inter-warp conflicts, as discussed in my previous article; negligible).

## 6. sgemm_tf32_swizzle_bcf_dbf

Now, add the double buffer pipeline to our conflict-free kernel, and we have the final, fully evolved kernel. Here's the complete benchmark and NCU summary:

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

![](../static/final.png)

From 15.925 ms to 8.276 ms — another hand-crafted performance optimization journey, going head-to-head with cuBLAS and coming out on top. Throughout this process, we used precise resource allocation, intricate coordinate mappings tailored to `ldmatrix`/`mma` (each different for A, B, and C fragments), grid swizzling to maximize L2 utilization, double buffer pipelining, and even reverse-engineered cuBLAS's own strategy to ultimately surpass it. Quite satisfying, isn't it?

## 7. Discussion

- Absolute benchmark numbers will fluctuate across runs (test device: RTX 5060 Laptop, subject to dynamic clocking, desktop/system background load). For narrative consistency, all results shown come from a single test session. Rest assured: while absolute timings vary, the final kernel **consistently beats cuBLAS** in every run.

- There was a painful detour during optimization: I spent significant time trying to transpose Bs via register-level shuffles. I got tunnel vision — determined to keep the Bs transpose. First I tried transposing during the write stage; then I tried loading `float2` and transposing in registers.
  - I did get it working (using four `__shfl_xor_sync` instructions), but performance wasn't great, so I abandoned it and only then reverse-engineered the cuBLAS shared-load strategy. The detour did deepen my understanding of warp shuffles, though.

- Why use grid swizzle for TF32 GEMM but not in the SIMT SGEMM? I actually tried it there too, but it had no effect. The reason: CUDA Core compute is slow enough that compute latency already masks memory latency.
  - With Tensor Cores, however, compute is much faster and can no longer hide memory latency — making L2 cache utilization critical.

- The final kernel consistently beats cuBLAS. Is there room for further optimization? Absolutely. For instance, due to `mma`'s rigid C fragment output layout, the C matrix write-back is not perfectly coalesced.
  - The standard solution is to use As/Bs SMEM as a staging buffer: rearrange data in SMEM first, then write coalesced to GMEM. But I'm out of energy for now — after all the fragment layouts, coordinate mappings, and swizzle derivations, my brain is overheating. Adding another SMEM pass means another round of conflict analysis. Since we already beat cuBLAS, I'll take a break.

- Circling back to the opening question: how can cuBLAS — with 0 bank conflicts, a 4-stage pipeline, and high L2 hit rates — be slower than our 2-stage pipeline?
  - First, I believe NVIDIA hasn't specifically optimized for consumer GPUs; this isn't their best kernel for this hardware.
  - Second, more pipeline stages aren't always better. The 4-stage pipeline requires more registers for scheduling (NCU shows 228), leaving only 1 active block per SM.
  - Our tiling strategy achieves a higher compute-to-memory ratio than cuBLAS:
    - We use 128×128 tiling; cuBLAS uses 64×256. Per-block data movement: 128+128 vs 64+256 units.
    - Our compute-to-memory ratio is (128×128 / (128+128)) / (64×256 / (64+256)) = **1.25× higher** than cuBLAS.
    - Combined with the 2-stage pipeline's lower register pressure, we can have 2 active blocks per SM, giving the scheduler more warps to hide memory latency.
  - If any NVIDIA experts happen to read this, I'd love to hear your take.

- Throughout TF32 `mma` development, I felt a persistent awkwardness: `ldmatrix` has no TF32 dtype and can't use the `.trans` modifier (`ldmatrix` operates on `.b16` data blocks, agnostic to higher-level types like FP16 or TF32).
  - The TF32 Tensor Core support introduced with Ampere is admittedly not as polished. But after this deep dive, writing HGEMM should be much smoother.

The discussion points above include some personal speculation — happy to hear other perspectives.

## 8. Conclusion

After this journey, we've developed a deeper understanding of Tensor Core instruction usage, warp-level cooperation, efficient SMEM utilization, and how swizzling and double buffering drive performance. While this article covers certain aspects of `mma`/`ldmatrix` in detail (e.g., the B fragment), it doesn't exhaustively explain everything — fragment layouts for A, B, and C differ across precision modes and shapes. For the complete picture, please refer to NVIDIA's official documentation and the PTX ISA manual.

Corrections and feedback are welcome. The complete kernel and test code are available on GitHub — check out my hand-crafted kernel series, the [**vitamin-cuda**](https://github.com/WingEdge777/Vitamin-CUDA) project.

That's all.
