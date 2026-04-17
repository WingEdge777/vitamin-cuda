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

// CTA 4D TMA: global -> shared
__device__ __forceinline__ void cp_async_bulk_tensor_4d(
    mbarrier_t *mbar, const void *tmap, const void *smem_ptr, int32_t s_0, int32_t s_1, int32_t s_2, int32_t s_3) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));

    asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3, %4, %5}], [%6];\n" ::"r"(smem_addr),
                 "l"(tmap),
                 "r"(s_0),
                 "r"(s_1),
                 "r"(s_2),
                 "r"(s_3),
                 "r"(mbar_addr)
                 : "memory");
}

template <const int BM, const int HEAD_DIM, typename T>
__device__ __forceinline__ void
ldmatrix_Qs(uint32_t reg_q[8][4], T (*Qs)[BM][HEAD_DIM / 2], int warp_row_offset, int lane_id) {
#pragma unroll
    for (int k_step = 0; k_step < 8; ++k_step) {
        int chunk = (k_step >= 4) ? 1 : 0;
        int local_k = (k_step % 4) * 16;
        int row = warp_row_offset + (lane_id % 16);
        uint32_t smem_addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&Qs[chunk][row][SWIZZLE_128B_TMA(row, local_k + (lane_id / 16) * 8)]));
        LDMATRIX_X4(reg_q[k_step][0], reg_q[k_step][1], reg_q[k_step][2], reg_q[k_step][3], smem_addr);
    }
}

template <const int BN, const int HEAD_DIM, typename T>
__device__ __forceinline__ void
ldmatrix_Ks(uint32_t reg_k[BN / 8][2], T (*Ks)[BN][HEAD_DIM / 2], int lane_id, int k_step) {
    int chunk = (k_step >= 4) ? 1 : 0;
    int local_k_row = (k_step % 4) * 16;

#pragma unroll
    for (int n_step = 0; n_step < BN / 8; ++n_step) {
        int k_row_in_smem = n_step * 8 + (lane_id % 8);
        int k_col_in_smem = local_k_row + ((lane_id % 16) / 8) * 8;

        uint32_t smem_addr_k = static_cast<uint32_t>(
            __cvta_generic_to_shared(&Ks[chunk][k_row_in_smem][SWIZZLE_128B_TMA(k_row_in_smem, k_col_in_smem)]));

        LDMATRIX_X2(reg_k[n_step][0], reg_k[n_step][1], smem_addr_k);
    }
}

template <const int BN, const int HEAD_DIM, typename T>
__device__ __forceinline__ void ldmatrix_Vs(uint32_t reg_v[16][2], T (*Vs)[BN][HEAD_DIM / 2], int lane_id, int k_step) {
#pragma unroll
    for (int n_step = 0; n_step < 16; ++n_step) {
        int chunk = (n_step >= 8) ? 1 : 0;
        int local_n = (n_step % 8) * 8;
        int v_row = k_step * 16 + (lane_id % 16);

        uint32_t smem_addr_v =
            static_cast<uint32_t>(__cvta_generic_to_shared(&Vs[chunk][v_row][SWIZZLE_128B_TMA(v_row, local_n)]));

        LDMATRIX_X2_TRANS(reg_v[n_step][0], reg_v[n_step][1], smem_addr_v);
    }
}

__device__ __forceinline__ uint32_t pack_bfloat2(float2 x) {
    union {
        __nv_bfloat162 bf16x2;
        uint32_t u32;
    } packed;
    packed.bf16x2 = __float22bfloat162_rn(x);
    return packed.u32;
}

template <const int STEPS>
__device__ __forceinline__ void mma_compute(float acc[STEPS][4], uint32_t reg_a[4], uint32_t reg_b[STEPS][2]) {
#pragma unroll
    for (int step = 0; step < STEPS; ++step) {
        M16N8K16_BF16(acc[step][0],
                      acc[step][1],
                      acc[step][2],
                      acc[step][3],
                      reg_a[0],
                      reg_a[1],
                      reg_a[2],
                      reg_a[3],
                      reg_b[step][0],
                      reg_b[step][1]);
    }
}

template <const int BM, const int HEAD_DIM, const int THREADS_PER_BLOCK = 128, typename T>
__device__ __forceinline__ void epilogue_writeback(float acc_o[16][4],
                                                   float m_i[2],
                                                   float d_i[2],
                                                   T (*Os)[HEAD_DIM],
                                                   T *o,
                                                   int warp_row_offset,
                                                   int lane_id,
                                                   int q_start_idx,
                                                   int q_len,
                                                   int q_head,
                                                   int batch_id,
                                                   int head_id) {

    int row_0 = warp_row_offset + (lane_id / 4);
    int row_1 = row_0 + 8;
    float inv_d0 = __frcp_rn(d_i[0]);
    float inv_d1 = __frcp_rn(d_i[1]);

#pragma unroll
    for (int n_step = 0; n_step < 16; ++n_step) {
        int col_base = n_step * 8 + (lane_id % 4) * 2;
        acc_o[n_step][0] *= inv_d0;
        acc_o[n_step][1] *= inv_d0;
        BFLOAT2(Os[row_0][SWIZZLE_128B_TMA(row_0, col_base)]) = __float22bfloat162_rn(FLOAT2(acc_o[n_step][0]));
    }

#pragma unroll
    for (int n_step = 0; n_step < 16; ++n_step) {
        int col_base = n_step * 8 + (lane_id % 4) * 2;
        acc_o[n_step][2] *= inv_d1;
        acc_o[n_step][3] *= inv_d1;
        BFLOAT2(Os[row_1][SWIZZLE_128B_TMA(row_1, col_base)]) = __float22bfloat162_rn(FLOAT2(acc_o[n_step][2]));
    }

    __syncthreads();

    const int CHUNKS_PER_ROW = HEAD_DIM / 8;
    const int row_stride = q_head * HEAD_DIM;
    const int block_base_idx = batch_id * (q_len * row_stride) + head_id * HEAD_DIM + q_start_idx * row_stride;

    const int TOTAL_CHUNKS = BM * CHUNKS_PER_ROW;
    const int ITERS = TOTAL_CHUNKS / THREADS_PER_BLOCK;
    const int tid = threadIdx.x;

#pragma unroll
    for (int iter = 0; iter < ITERS; ++iter) {
        int i = tid + iter * THREADS_PER_BLOCK;
        int r = i / CHUNKS_PER_ROW;
        int c = (i % CHUNKS_PER_ROW) * 8;

        if (q_start_idx + r < q_len) {
            int global_idx = block_base_idx + r * row_stride + c;
            LDST128BITS(o[global_idx]) = LDST128BITS(Os[r][SWIZZLE_128B_TMA(r, c)]);
        }
    }
}

__device__ __forceinline__ void lazy_rescale(float acc_o[16][4], float d_i[2], float row_scale[2], int row_idx) {
    if (row_scale[row_idx] == 1.0f) {
        return;
    }

    float scale = row_scale[row_idx];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        acc_o[i][row_idx * 2 + 0] *= scale;
        acc_o[i][row_idx * 2 + 1] *= scale;
    }
    d_i[row_idx] *= scale;
    row_scale[row_idx] = 1.0f;
}

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

    // mbar array at end of SMEM (8-byte aligned;
    mbarrier_t *mbar_q = reinterpret_cast<mbarrier_t *>(smem_buf + (BM + BN * 2) * HEAD_DIM * sizeof(T));
    mbarrier_t *mbar_k =
        reinterpret_cast<mbarrier_t *>(smem_buf + (BM + BN * 2) * HEAD_DIM * sizeof(T) + sizeof(mbarrier_t));
    mbarrier_t *mbar_v = mbar_k + 1;

    // 2. coordinates
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
        // 2 TMA instructions to load the 128-dim split into left and right halves
        cp_async_bulk_tensor_4d(mbar_q, &tma_q, Qs[0], 0, head_id, q_start_idx, batch_id);
        cp_async_bulk_tensor_4d(mbar_q, &tma_q, Qs[1], 64, head_id, q_start_idx, batch_id);
    }
    __syncthreads();
    mbarrier_wait(mbar_q, 0); // Wait for Q to finish loading (Q is used throughout the inner loop)

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warp_row_offset = warp_id * 16;

    // 4. ldmatrix q
    uint32_t reg_q[8][4];
    ldmatrix_Qs<BM, HEAD_DIM>(reg_q, Qs, warp_row_offset, lane_id);

    // 5. Initialize output registers
    float acc_o[16][4] = {0.0f};

    // m_i and d_i track max value and softmax denominator for each of the two rows per thread
    float m_i[2] = {-FLT_MAX, -FLT_MAX};
    float d_i[2] = {0.0f, 0.0f};
    float row_scale[2] = {1.0f, 1.0f};

    const int k_end = min(kv_len, q_start_idx + BM); // Causal mask boundary, upper triangle is all zeros so skip
    int phase_k = 0;
    int phase_v = 0;
    const float scale_log2 = scale * 1.44269504f; // scale*log2(e)
    // P is written out in BF16, so avoid letting the deferred renorm grow beyond
    // roughly one BF16 mantissa's worth of amplification (2^8). Once row_scale
    // drops below 2^-8, materialize it back into acc_o/d_i before quantizing P.
    constexpr float lazy_scale_threshold = 0x1p-8f;

    // 6. kv loop
    for (int n = 0; n < k_end; n += BN) {
        // --- 6.1 TMA async load KV ---
        if (tid == 0) {
            mbarrier_expect_tx(mbar_k, BN * HEAD_DIM * sizeof(T));
            mbarrier_expect_tx(mbar_v, BN * HEAD_DIM * sizeof(T));
            cp_async_bulk_tensor_4d(mbar_k, &tma_k, Ks[0], 0, kv_head_id, n, batch_id);
            cp_async_bulk_tensor_4d(mbar_k, &tma_k, Ks[1], 64, kv_head_id, n, batch_id);
            cp_async_bulk_tensor_4d(mbar_v, &tma_v, Vs[0], 0, kv_head_id, n, batch_id);
            cp_async_bulk_tensor_4d(mbar_v, &tma_v, Vs[1], 64, kv_head_id, n, batch_id);
        }
        __syncthreads();
        mbarrier_wait(mbar_k, phase_k);
        phase_k ^= 1; // flip phase

        // --- 6.2 Compute S = Q * K^T ---
        float acc_s[8][4] = {0.f};

#pragma unroll
        for (int k_step = 0; k_step < 8; ++k_step) {
            uint32_t reg_k[8][2];
            ldmatrix_Ks<BN, HEAD_DIM>(reg_k, Ks, lane_id, k_step);
            mma_compute<8>(acc_s, reg_q[k_step], reg_k);
        }

        // --- 6.3 Online Softmax (Max & Correction) ---
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

        // 6.4 Normalize P in registers and accumulate local denominator
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
            uint32_t reg_p[4];
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

#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

template <typename T, const int rowBytes = 128>
inline CUtensorMap create_4d_tensor_map(T *global_address,
                                        uint64_t dim_d,
                                        uint64_t dim_h,
                                        uint64_t dim_s,
                                        uint64_t dim_b,
                                        uint64_t stride_h,
                                        uint64_t stride_s,
                                        uint64_t stride_b, // Byte stride
                                        uint32_t box_d,
                                        uint32_t box_s) // Each kernel load takes a (box_s x box_d) block
{
    CUtensorMap tmap;
    CUtensorMapDataType type =
        std::is_same_v<T, __half> ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    CUtensorMapSwizzle swizzle = rowBytes == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B;

    // TMA dimensions: from fastest (0) to slowest (3)
    // Assuming memory layout [Batch, Seq, Head, Dim], Dim is fastest
    uint64_t globalDim[4] = {dim_d, dim_h, dim_s, dim_b};

    // globalStrides are strides for dimensions 1, 2, 3, must be in Bytes
    uint64_t globalStrides[3] = {stride_h, stride_s, stride_b};

    // boxDim is the size we load at once for each dimension
    // For Head and Batch dimensions, we only load one at a time (hence 1)
    uint32_t boxDim[4] = {box_d, 1, box_s, 1};
    uint32_t elementStrides[4] = {1, 1, 1, 1};

    CUresult res = cuTensorMapEncodeTiled(&tmap,
                                          type,
                                          4, // Rank = 4
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
        /* Extract dimension info dynamically from Tensor */                                                           \
        const int batch_size = q.size(0);                                                                              \
        const int q_len = q.size(1);                                                                                   \
        const int q_head = q.size(2);                                                                                  \
        const int head_dim = q.size(3);                                                                                \
        const int kv_len = k.size(1);                                                                                  \
        const int kv_head = k.size(2);                                                                                 \
                                                                                                                       \
        /* Only validate that head_dim matches the compile-time constant */                                            \
        TORCH_CHECK(head_dim == HEAD_DIM, "Head dim mismatch: expected ", HEAD_DIM);                                   \
                                                                                                                       \
        /* Extract byte strides dynamically (TMA requires byte counts) */                                              \
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
        /* q_seq on x-dimension to reuse L2 cache for KV tiles */                                                      \
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

binding_tiled_tma_func_gen(fmha_tma, 128);

#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // fmha_tma_128
    torch_pybinding_func(fmha_tma_128);
}
