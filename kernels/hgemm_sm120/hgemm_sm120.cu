#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/types.h>

#include "../common/pack.cuh"

#define SWIZZLE_A(row, col) ((col) ^ (((row >> 1) & 0x3) << 3))

#define SWIZZLE_B(row, col) ((col) ^ (((row) & 0x7) << 3))

#define SWIZZLE_C(row, col) ((col) ^ (((row) & 0x7) << 3))

#define SWIZZLE_64B_TMA(row, col) ((col) ^ (((row >> 1) & 0x3) << 3))

#define SWIZZLE_128B_TMA(row, col) ((col) ^ (((row) & 0x7) << 3))

// ---------------- Inline PTX assembly macros ----------------
// cp.async: async 16-byte copy from gmem (src) to smem (dst_smem_32b)
#define CP_ASYNC_CG(dst_smem_32b, src_global_ptr)                                                                      \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(dst_smem_32b), "l"(src_global_ptr))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    if constexpr (N == 0)
        asm volatile("cp.async.wait_group 0;\n" ::);
    else if constexpr (N == 1)
        asm volatile("cp.async.wait_group 1;\n" ::);
    else if constexpr (N == 2)
        asm volatile("cp.async.wait_group 2;\n" ::);
    else if constexpr (N == 3)
        asm volatile("cp.async.wait_group 3;\n" ::);
}

// ldmatrix
#define LDMATRIX_X4(R0, R1, R2, R3, PTR)                                                                               \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"                                    \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                                              \
                 : "r"(PTR))

#define LDMATRIX_X2(R0, R1, PTR)                                                                                       \
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(R0), "=r"(R1) : "r"(PTR))

// Load two 8x8 tiles and transpose
#define LDMATRIX_X2_TRANS(R0, R1, PTR)                                                                                 \
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];" : "=r"(R0), "=r"(R1) : "r"(PTR))

// mma.sync
#define M16N8K16_F16(C0, C1, C2, C3, A0, A1, A2, A3, B0, B1)                                                           \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "                                                  \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"                                         \
                 : "=f"(C0), "=f"(C1), "=f"(C2), "=f"(C3)                                                              \
                 : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1), "f"(C0), "f"(C1), "f"(C2), "f"(C3))

#define M16N8K16_BF16(C0, C1, C2, C3, A0, A1, A2, A3, B0, B1)                                                          \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "                                                \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"                                         \
                 : "=f"(C0), "=f"(C1), "=f"(C2), "=f"(C3)                                                              \
                 : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1), "f"(C0), "f"(C1), "f"(C2), "f"(C3))

// ---------------- MBarrier & TMA (pure PTX helpers) ----------------

// MBarrier object (8-byte aligned per hardware)
typedef uint64_t mbarrier_t;

// Init barrier (single thread in prologue)
__device__ __forceinline__ void mbarrier_init(mbarrier_t *mbar, uint32_t expected_count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar))),
                 "r"(expected_count));
}

// Expected byte count for the TMA transfer
__device__ __forceinline__ void mbarrier_expect_tx(mbarrier_t *mbar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(
                     static_cast<uint32_t>(__cvta_generic_to_shared(mbar))),
                 "r"(tx_bytes));
}

// Consumer-side arrive (flip phase) after draining smem — unused here: no producer/consumer warp specialied
// warp 0 issues loads while all warps consume; __syncthreads() is enough for our pattern.
__device__ __forceinline__ void mbarrier_arrive(mbarrier_t *mbar) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar))));
}

// Wait until the TMA copy completes
__device__ __forceinline__ void mbarrier_wait(uint64_t *smem_ptr, uint32_t phase) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    // Large spin timeout (0x989680 == 10,000,000 cycles)
    uint32_t ticks = 0x989680;

    asm volatile("{\n\t"
                 ".reg .pred p; \n\t"
                 "LAB_WAIT: \n\t"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2; \n\t"
                 "@p bra DONE; \n\t"
                 "bra LAB_WAIT; \n\t"
                 "DONE: \n\t"
                 "}\n"
                 :
                 : "r"(smem_addr), "r"(phase), "r"(ticks)
                 : "memory");
}

// CTA 2D TMA: global -> shared
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

const int WARP_SIZE = 32;

template <const int BK, typename T>
__device__ __forceinline__ void
ldmatrix_A(uint32_t reg_a[4][4], T (*As)[BK], int warp_id_m, int lane_id, int k_offset) {

    // Four ldmatrix.issue for A (4 * 16 = 64 rows)
#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
        // ldmatrix x4 loads a 16x16 tile
        int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
        int a_col = k_offset + (lane_id / 16) * 8;
        uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&As[a_row][SWIZZLE_A(a_row, a_col)]));
        LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
    }
}

template <const int BK, typename T>
__device__ __forceinline__ void
ldmatrix_A_tma(uint32_t reg_a[4][4], T (*As)[BK], int warp_id_m, int lane_id, int k_offset) {

    // Four ldmatrix.issue for A (4 * 16 = 64 rows)
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
ldmatrix_B(uint32_t reg_b[4][2], T (*Bs)[BN], int warp_id_n, int lane_id, int k_offset) {

    // Four ldmatrix.issue for B (4 * 8 = 32 columns)
#pragma unroll
    for (int n_idx = 0; n_idx < 4; ++n_idx) {
        // Lanes 0-15 load 16 rows (bases of two 8x8 tiles)
        int b_row = k_offset + (lane_id % 16);
        int b_col = warp_id_n * 32 + n_idx * 8;

        uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[b_row][SWIZZLE_B(b_row, b_col)]));
        LDMATRIX_X2_TRANS(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
    }
}

template <const int BN, const int BK, typename T>
__device__ __forceinline__ void
ldmatrix_B_tma(uint32_t reg_b[4][2], T (*Bs)[BK][BN / 2], int warp_id_n, int lane_id, int k_offset) {
#pragma unroll
    for (int n_idx = 0; n_idx < 4; ++n_idx) {
        int b_row = k_offset + (lane_id % 16);
        int b_col = warp_id_n * 32 + n_idx * 8;

        // Chunk-dependent indexing
        int chunk_idx = b_col / (BN / 2);
        int local_col = b_col % (BN / 2);

        uint32_t smem_addr =
            static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[chunk_idx][b_row][SWIZZLE_128B_TMA(b_row, local_col)]));
        LDMATRIX_X2_TRANS(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
    }
}

template <typename T>
__device__ __forceinline__ void mma_compute(float sum[4][4][4], uint32_t reg_a[4][4], uint32_t reg_b[4][2]) {

    // MMA body: 4x4 m16n8k16
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

template <const int BK, typename T>
__device__ __forceinline__ void cp_async_load_A(T (*As)[BK], int load_a_row, int load_a_col, T *global_a_ptr, int k) {

    // cp.async A: each thread 2 rows, spanning 64 rows
    uint32_t smem_a0 =
        static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row][SWIZZLE_A(load_a_row, load_a_col)]));
    uint32_t smem_a1 =
        static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row + 64][SWIZZLE_A(load_a_row + 64, load_a_col)]));
    // A tile spans 64 rows
    CP_ASYNC_CG(smem_a0, global_a_ptr);
    CP_ASYNC_CG(smem_a1, global_a_ptr + 64 * k);
}

template <const int BK, const int BN, typename T>
__device__ __forceinline__ void cp_async_load_B(T (*Bs)[BN], int load_b_row, int load_b_col, T *global_b_ptr, int n) {

    // cp.async B: each thread 2 rows, spanning 16 rows
    uint32_t smem_b0 =
        static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[load_b_row][SWIZZLE_B(load_b_row, load_b_col)]));
    uint32_t smem_b1 =
        static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[load_b_row + 16][SWIZZLE_B(load_b_row + 16, load_b_col)]));
    // B tile spans 16 rows
    CP_ASYNC_CG(smem_b0, global_b_ptr);
    CP_ASYNC_CG(smem_b1, global_b_ptr + 16 * n);
}

template <const int BM, const int BN, typename T>
__device__ __forceinline__ void write_c_via_smem(
    T *c, int by, int bx, int n, float sum[4][4][4], int warp_id_m, int warp_id_n, int lane_id, int tid, T (*Cs)[BN]) {

    // ---------------- Store C ----------------
    // Reuse As/Bs smem as staging for C stores
    __syncthreads();

    int t_row = lane_id / 4;       // 0~7
    int t_col = (lane_id % 4) * 2; // 0, 2, 4, 6

    // register to Cs smem
#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int c_base_row = warp_id_m * 64 + m_idx * 16; // M: 16-row span
            int c_base_col = warp_id_n * 32 + n_idx * 8;  // N: 8-column span

            // Store 16 rows in two 8-row passes
            int c_row_0 = c_base_row + t_row;
            int c_row_2 = c_base_row + t_row + 8;
            int c_col = c_base_col + t_col;

            if constexpr (std::is_same_v<T, __half>) {
                HALF2(Cs[c_row_0][SWIZZLE_C(c_row_0, c_col)]) = __float22half2_rn(FLOAT2(sum[m_idx][n_idx][0]));
                HALF2(Cs[c_row_2][SWIZZLE_C(c_row_2, c_col)]) = __float22half2_rn(FLOAT2(sum[m_idx][n_idx][2]));
            } else {
                BFLOAT2(Cs[c_row_0][SWIZZLE_C(c_row_0, c_col)]) = __float22bfloat162_rn(FLOAT2(sum[m_idx][n_idx][0]));
                BFLOAT2(Cs[c_row_2][SWIZZLE_C(c_row_2, c_col)]) = __float22bfloat162_rn(FLOAT2(sum[m_idx][n_idx][2]));
            }
        }
    }

    __syncthreads();

    // smem to gmem
    // Each thread moves 64 elems (fp16/bf16) == 8 float4; one warp moves 32*4*4 = 512 B; 256 threads -> 4096 B
    T *c_block = &c[by * BM * n + bx * BN];

#pragma unroll
    for (int step = 0; step < 8; ++step) {
        // Keep elem_idx contiguous within the warp (coalesced)
        int elem_idx = (step * 256 + tid) * 8;
        int row = elem_idx / 128;
        int col = elem_idx % 128;

        int s_col = SWIZZLE_C(row, col);

        FLOAT4(c_block[row * n + col]) = FLOAT4(Cs[row][s_col]);
    }
}

// ------------------------------------------ cp.async + mma  ----------------------------------------------------

// a block calculate c[128][128]
template <const int BM = 128, const int BN = 128, const int BK = 32, typename T>
__global__ void hgemm_bcf_dbf_rw_kernel(T *a, T *b, T *c, int m, int n, int k) {
    // grid swizzling
    int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8; // swizzle / logical width = 8

    int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    int by = (linear_id / SWIZZLE_W) % gridDim.y;

    int tid = threadIdx.x; // 0~255
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // GMEM->SMEM load indexing
    int load_a_row = tid / 4;        // 0~63
    int load_a_col = (tid % 4) * 8;  // 0,8,16,24
    int load_b_row = tid / 16;       // 0..15 (K)
    int load_b_col = (tid % 16) * 8; // 0,8,...,120 (N)

    // Layout works for both A and B; union aliases one smem block for A/B then C
    __shared__ __align__(128) union {
        // First phase: A and B tiles
        struct {
            T As[2][BM][BK];
            T Bs[2][BK][BN];
        };
        // Second phase: C writeback buffer
        T Cs[BM][BN];
    } smem;

    // warp tiling
    // Each warp owns a 64x32 C tile
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // Accumulators: 4 M-fragments * 4 N-fragments * 4 float regs = 64
    float sum[4][4][4] = {0.f};

    T *global_a_ptr = &a[(by * BM + load_a_row) * k + load_a_col];
    T *global_b_ptr = &b[load_b_row * n + bx * BN + load_b_col];

    // ----------------------------- Prologue: prefetch As/Bs once
    // Strided loader covers full 128x32 / 32x128 tiles
    cp_async_load_A<BK>(smem.As[0], load_a_row, load_a_col, global_a_ptr, k);
    cp_async_load_B<BK, BN>(smem.Bs[0], load_b_row, load_b_col, global_b_ptr, n);

    CP_ASYNC_COMMIT_GROUP();
    cp_async_wait_group<0>();
    __syncthreads();

    int read_idx = 0;
    int write_idx = 1;

    // Main K loop
    for (int bk = 32; bk < k; bk += BK) {

        // Advance source pointers
        global_a_ptr += BK;
        global_b_ptr += BK * n;

        // 1. cp.async load A/B
        cp_async_load_A<BK>(smem.As[write_idx], load_a_row, load_a_col, global_a_ptr, k);
        cp_async_load_B<BK, BN>(smem.Bs[write_idx], load_b_row, load_b_col, global_b_ptr, n);

        CP_ASYNC_COMMIT_GROUP();

        // 2. Tensor Core: two K steps, 16 elements each
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 16;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // Four ldmatrix.issue for A (4 * 16 = 64 rows)
            ldmatrix_A<BK>(reg_a, smem.As[read_idx], warp_id_m, lane_id, k_offset);

            // Four ldmatrix.issue for B (4 * 8 = 32 columns)
            ldmatrix_B<BN, BK>(reg_b, smem.Bs[read_idx], warp_id_n, lane_id, k_offset);

            // MMA body: 4x4 m16n8k16
            mma_compute<T>(sum, reg_a, reg_b);
        }

        read_idx ^= 1;
        write_idx ^= 1;

        cp_async_wait_group<0>();
        __syncthreads();
    }
    // ------------------- Epilogue: final MMA, then store C
#pragma unroll
    for (int k_step = 0; k_step < 2; ++k_step) {
        int k_offset = k_step * 16;

        uint32_t reg_a[4][4];
        uint32_t reg_b[4][2];

        // Four ldmatrix.issue for A (4 * 16 = 64 rows)
        ldmatrix_A<BK>(reg_a, smem.As[read_idx], warp_id_m, lane_id, k_offset);

        // Four ldmatrix.issue for B (4 * 8 = 32 columns)
        ldmatrix_B<BN, BK>(reg_b, smem.Bs[read_idx], warp_id_n, lane_id, k_offset);

        // MMA body: 4x4 m16n8k16
        mma_compute<T>(sum, reg_a, reg_b);
    }

    write_c_via_smem<BM, BN>(c, by, bx, n, sum, warp_id_m, warp_id_n, lane_id, tid, smem.Cs);
}

template <const int BM = 128, const int BN = 128, const int BK = 32, const int STAGES = 3, typename T>
__global__ void hgemm_k_stages_kernel(T *a, T *b, T *c, int m, int n, int k) {
    // grid swizzling
    int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8; // swizzle / logical width = 8

    int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    int by = (linear_id / SWIZZLE_W) % gridDim.y;

    int tid = threadIdx.x; // 0~255
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // GMEM->SMEM load indexing
    int load_a_row = tid / 4;        // 0~63
    int load_a_col = (tid % 4) * 8;  // 0,8,16,24
    int load_b_row = tid / 16;       // 0..15 (K)
    int load_b_col = (tid % 16) * 8; // 0,8,...,120 (N)

    // Layout works for both A and B; union aliases one smem block for A/B then C
    __shared__ __align__(128) union {
        // First phase: A and B tiles
        struct {
            T As[STAGES][BM][BK];
            T Bs[STAGES][BK][BN];
        };
        // Second phase: C writeback buffer
        T Cs[BM][BN];
    } smem;

    // warp tiling
    // Each warp owns a 64x32 C tile
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // Accumulators: 4 M-fragments * 4 N-fragments * 4 float regs = 64
    float sum[4][4][4] = {0.f};

    T *global_a_ptr = &a[(by * BM + load_a_row) * k + load_a_col];
    T *global_b_ptr = &b[load_b_row * n + bx * BN + load_b_col];

    // 1. Prologue: prefetch STAGES-1 As/Bs blocks
#pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) {
        cp_async_load_A<BK>(smem.As[i], load_a_row, load_a_col, global_a_ptr, k);
        cp_async_load_B<BK, BN>(smem.Bs[i], load_b_row, load_b_col, global_b_ptr, n);

        CP_ASYNC_COMMIT_GROUP();

        global_a_ptr += BK;
        global_b_ptr += BK * n;
    }
    // Two commit groups: one may still be completing cp.async while the earliest load finishes
    cp_async_wait_group<STAGES - 2>();
    __syncthreads();

    // Pipeline state init
    int load_stage = STAGES - 1; // next stage to fill
    int compute_stage = 0;       // stage currently consumed by MMA

    // 2. main loop
    for (int bk = (STAGES - 1) * BK; bk < k; bk += BK) {

        // 1. Issue cp.async loads into load_stage
        cp_async_load_A<BK>(smem.As[load_stage], load_a_row, load_a_col, global_a_ptr, k);
        cp_async_load_B<BK, BN>(smem.Bs[load_stage], load_b_row, load_b_col, global_b_ptr, n);

        CP_ASYNC_COMMIT_GROUP();

        // 2. Tensor Core: two K steps, 16 elements each
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 16;
            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // Four ldmatrix.issue for A (4 * 16 = 64 rows)
            ldmatrix_A<BK>(reg_a, smem.As[compute_stage], warp_id_m, lane_id, k_offset);

            // Four ldmatrix.issue for B (4 * 8 = 32 columns)
            ldmatrix_B<BN, BK>(reg_b, smem.Bs[compute_stage], warp_id_n, lane_id, k_offset);

            // MMA body: 4x4 m16n8k16
            mma_compute<T>(sum, reg_a, reg_b);
        }

        // Advance source pointers
        global_a_ptr += BK;
        global_b_ptr += BK * n;

        // Advance pipeline by one stage
        load_stage = (load_stage + 1 == STAGES) ? 0 : load_stage + 1;
        compute_stage = (compute_stage + 1 == STAGES) ? 0 : compute_stage + 1;

        // Wait until the oldest group's loads complete
        cp_async_wait_group<STAGES - 2>();
        __syncthreads();
    }

    // 3. Epilogue: STAGES-1 remaining passes, then store
    cp_async_wait_group<0>();
    __syncthreads();
#pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) {
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 16;
            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // Four ldmatrix.issue for A (4 * 16 = 64 rows)
            ldmatrix_A<BK>(reg_a, smem.As[compute_stage], warp_id_m, lane_id, k_offset);

            // Four ldmatrix.issue for B (4 * 8 = 32 columns)
            ldmatrix_B<BN, BK>(reg_b, smem.Bs[compute_stage], warp_id_n, lane_id, k_offset);

            // MMA body: 4x4 m16n8k16
            mma_compute<T>(sum, reg_a, reg_b);
        }

        compute_stage = (compute_stage + 1 == STAGES) ? 0 : compute_stage + 1;
    }

    write_c_via_smem<BM, BN>(c, by, bx, n, sum, warp_id_m, warp_id_n, lane_id, tid, smem.Cs);
}

// -------------------   tma r + mma -------------------
// a block calculate c[128][128]
template <const int BM = 128, const int BN = 128, const int BK = 64, const int STAGES = 3, typename T>
__global__ void hgemm_tma_r_k_stages_kernel(
    __grid_constant__ const CUtensorMap tma_a, __grid_constant__ const CUtensorMap tma_b, T *c, int m, int n, int k) {
    // grid swizzling
    int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8; // swizzle / logical width = 8

    int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    int by = (linear_id / SWIZZLE_W) % gridDim.y;

    int tid = threadIdx.x; // 0~255
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // Dynamic shared memory
    extern __shared__ __align__(128) uint8_t smem_buf[];
    T(*As)[BM][BK] = reinterpret_cast<T(*)[BM][BK]>(smem_buf);
    T(*Bs)[2][BK][BN / 2] = reinterpret_cast<T(*)[2][BK][BN / 2]>(smem_buf + STAGES * BM * BK * sizeof(T));
    T(*Cs)[BN] = reinterpret_cast<T(*)[BN]>(smem_buf);
    // mbar array at end of SMEM (8-byte aligned; one per stage)
    mbarrier_t *mbar = reinterpret_cast<mbarrier_t *>(smem_buf + BM * BK * sizeof(T) * STAGES * 2);

    // Init MBarrier (tid 0 only; expected arrivals == 1 from TMA)
    if (tid == 0) {
        for (int i = 0; i < STAGES; ++i)
            mbarrier_init(&mbar[i], 1);
    }
    __syncthreads(); // ensure mbar init is visible

    // warp tiling
    // Each warp owns a 64x32 C tile
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // Accumulators: 4 M-fragments * 4 N-fragments * 4 float regs = 64
    float sum[4][4][4] = {0.f};

    // Bytes moved per TMA descriptor
    const uint32_t tx_bytes = (BM * BK + BK * BN) * sizeof(T);

    // Simple tile counter only (strides live in the TMA descriptor)
    int load_k_coord = 0;

    // 1. Prologue: prefetch STAGES - 1 tiles
    for (int i = 0; i < STAGES - 1; ++i) {
        if (tid == 0) {
            // Bytes this mbarrier must observe before completion
            mbarrier_expect_tx(&mbar[i], tx_bytes);

            cp_async_bulk_tensor_2d(&mbar[i], &tma_a, As[i], load_k_coord, by * BM);
            // Bs needs two TMA tiles
            cp_async_bulk_tensor_2d(&mbar[i], &tma_b, Bs[i][0], bx * BN, load_k_coord);
            cp_async_bulk_tensor_2d(&mbar[i], &tma_b, Bs[i][1], bx * BN + BN / 2, load_k_coord);
        }
        load_k_coord += BK;
    }

    int load_stage = STAGES - 1;
    int compute_stage = 0;
    int wait_phase = 0; // MBarrier phase bit (toggles 0/1)
    int total_k_step = BK / 16;
    // 2. main loop
    for (int bk = (STAGES - 1) * BK; bk < k; bk += BK) {

        // Launch the next TMA (still only tid 0)
        if (tid == 0) {
            mbarrier_expect_tx(&mbar[load_stage], tx_bytes);
            cp_async_bulk_tensor_2d(&mbar[load_stage], &tma_a, As[load_stage], load_k_coord, by * BM);
            cp_async_bulk_tensor_2d(&mbar[load_stage], &tma_b, Bs[load_stage][0], bx * BN, load_k_coord);
            cp_async_bulk_tensor_2d(&mbar[load_stage], &tma_b, Bs[load_stage][1], bx * BN + BN / 2, load_k_coord);
        }
        load_k_coord += BK;

        // All threads spin until TMA finishes the current compute_stage tile
        mbarrier_wait(&mbar[compute_stage], wait_phase);

        // Register double-buffer: ldmatrix + MMA
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

        // Whole-CTA sync; no producer warp arrive()
        __syncthreads();

        // Rotate pipeline indices
        load_stage = (load_stage + 1 == STAGES) ? 0 : load_stage + 1;
        compute_stage = (compute_stage + 1 == STAGES) ? 0 : compute_stage + 1;

        // Full 3-stage lap: all mbar phases flipped; toggle wait_phase too
        if (compute_stage == 0)
            wait_phase ^= 1;
    }
    // 3. Epilogue: STAGES-1 passes
#pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) {
        // Wait on TMA again
        mbarrier_wait(&mbar[compute_stage], wait_phase);

        // Register double-buffer
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

    // 4. Store C
    write_c_via_smem<BM, BN>(c, by, bx, n, sum, warp_id_m, warp_id_n, lane_id, tid, Cs);
}

// -------------------  tma rw + mma -------------------
// a block calculate c[128][128]
template <const int BM = 128, const int BN = 128, const int BK = 32, const int STAGES = 3, typename T>
__global__ void hgemm_tma_rw_k_stages_kernel(T *a, T *b, T *c, int m, int n, int k) {}

#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

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
        if (a.dtype() == torch::kHalf) {                                                                               \
            name##_kernel<128, 128, 32>                                                                                \
                <<<blocks_per_grid, threads_per_block, 0, stream>>>(reinterpret_cast<__half *>(a.data_ptr()),          \
                                                                    reinterpret_cast<__half *>(b.data_ptr()),          \
                                                                    reinterpret_cast<__half *>(c.data_ptr()),          \
                                                                    M,                                                 \
                                                                    N,                                                 \
                                                                    K);                                                \
        } else {                                                                                                       \
            name##_kernel<128, 128, 32>                                                                                \
                <<<blocks_per_grid, threads_per_block, 0, stream>>>(reinterpret_cast<__nv_bfloat16 *>(a.data_ptr()),   \
                                                                    reinterpret_cast<__nv_bfloat16 *>(b.data_ptr()),   \
                                                                    reinterpret_cast<__nv_bfloat16 *>(c.data_ptr()),   \
                                                                    M,                                                 \
                                                                    N,                                                 \
                                                                    K);                                                \
        }                                                                                                              \
    }

template <typename T, const int rowBytes = 128>
inline CUtensorMap
create_tensor_map(T *global_address, uint64_t fast_dim, uint64_t slow_dim, uint32_t fast_box, uint32_t slow_box) {
    CUtensorMap tmap;
    CUtensorMapDataType type =
        std::is_same_v<T, __half> ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    CUtensorMapSwizzle swizzle = rowBytes == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B;

    // TMA convention: dim 0 is the fastest-changing dimension in memory
    uint64_t globalDim[2] = {fast_dim, slow_dim};
    uint64_t globalStrides[1] = {fast_dim * sizeof(T)}; // outer-dim stride (bytes)
    uint32_t boxDim[2] = {fast_box, slow_box};
    uint32_t elementStrides[2] = {1, 1};

    CUresult res = cuTensorMapEncodeTiled(&tmap,
                                          type,
                                          2, // tensor rank (2D matrix)
                                          global_address,
                                          globalDim,
                                          globalStrides,
                                          boxDim,
                                          elementStrides,
                                          CU_TENSOR_MAP_INTERLEAVE_NONE,
                                          swizzle, // swizzle mode
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

binding_tiled_func_gen(hgemm_bcf_dbf_rw);
binding_tiled_func_gen(hgemm_k_stages);

binding_tiled_tma_func_gen(hgemm_tma_r_k_stages, 64);
binding_tiled_tma_func_gen(hgemm_tma_r_k_stages, 32);

extern void hgemm_cublas(torch::Tensor a, torch::Tensor b, torch::Tensor c);

// binding
#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(hgemm_cublas);
    torch_pybinding_func(hgemm_bcf_dbf_rw);
    torch_pybinding_func(hgemm_k_stages);
    torch_pybinding_func(hgemm_tma_r_k_stages_64);
    torch_pybinding_func(hgemm_tma_r_k_stages_32);
}
