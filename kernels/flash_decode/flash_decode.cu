#include <cfloat>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <torch/types.h>

#include "../common/pack.cuh"

// ---------------- Inline PTX assembly macros ----------------
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

// Wait until the TMA copy completes.
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

// CTA 3D TMA: global -> shared
__device__ __forceinline__ void cp_async_bulk_tensor_3d(
    mbarrier_t *mbar, const void *tmap, const void *smem_ptr, int32_t s_0, int32_t s_1, int32_t s_2) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));

    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3, %4}], [%5];\n" ::"r"(smem_addr),
                 "l"(tmap),
                 "r"(s_0),
                 "r"(s_1),
                 "r"(s_2),
                 "r"(mbar_addr)
                 : "memory");
}

template <const int group_size>
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = group_size >> 1; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset, group_size);
    }
    return val;
}

template <const int group_size>
__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
    for (int offset = group_size >> 1; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset, group_size));
    }
    return val;
}

template <const int num_group, const int group_size>
__device__ __forceinline__ float block_reduce_sum(float val) {
    val = warp_reduce_sum<group_size>(val);

    __shared__ float sdata[num_group];
    const int lane_id = threadIdx.x % group_size;
    const int group_id = threadIdx.x / group_size;
    if (lane_id == 0) {
        sdata[group_id] = val;
    }
    __syncthreads();
    if (threadIdx.x < WARP_SIZE) {
        val = threadIdx.x < num_group ? sdata[threadIdx.x] : 0.0f;
        val = warp_reduce_sum<WARP_SIZE>(val);
        if (threadIdx.x == 0) {
            sdata[0] = val;
        }
    }
    __syncthreads();
    return sdata[0];
}

template <const int num_group, const int group_size>
__device__ __forceinline__ float block_reduce_max(float val) {
    val = warp_reduce_max<group_size>(val);

    __shared__ float sdata[num_group];
    const int lane_id = threadIdx.x % group_size;
    const int group_id = threadIdx.x / group_size;
    if (lane_id == 0) {
        sdata[group_id] = val;
    }
    __syncthreads();
    if (threadIdx.x < WARP_SIZE) {
        val = threadIdx.x < num_group ? sdata[threadIdx.x] : -FLT_MAX;
        val = warp_reduce_max<WARP_SIZE>(val);
        if (threadIdx.x == 0) {
            sdata[0] = val;
        }
    }
    __syncthreads();
    return sdata[0];
}

template <const int num_group, const int group_size>
__device__ __forceinline__ float block_reduce_sum_by_lane(float val) {
    __shared__ float sdata[num_group][group_size];
    const int lane_id = threadIdx.x % group_size;
    const int group_id = threadIdx.x / group_size;

    sdata[group_id][lane_id] = val;
    __syncthreads();
    if (group_id == 0) {
        val = 0.0f;
#pragma unroll
        for (int group = 0; group < num_group; ++group) {
            val += sdata[group][lane_id];
        }
        sdata[0][lane_id] = val;
    }
    __syncthreads();
    return sdata[0][lane_id];
}

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

// flash decoding softmax(q @k.T *scale) @v
template <const int BN = 64,
          const int CHUNK_SIZE = 256,
          const int HEAD_DIM = 128,
          const int THREADS_PER_BLOCK = 128,
          typename T>
__global__ void flash_decode_tma_dbf_k_kernel(T *q,
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
    T(*Ks)[BN][HEAD_DIM] = reinterpret_cast<T(*)[BN][HEAD_DIM]>(smem_buf);
    T(*Vs)[HEAD_DIM] = reinterpret_cast<T(*)[HEAD_DIM]>(smem_buf + BN * HEAD_DIM * sizeof(T) * 2);
    mbarrier_t *mbar_k = reinterpret_cast<mbarrier_t *>(smem_buf + BN * HEAD_DIM * sizeof(T) * 3);
    mbarrier_t *mbar_v = mbar_k + 2;

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

    // 3. load q fragment
    pack128 qs{FLOAT4(q[q_head_id * HEAD_DIM + lane_id * 8])};

    // 4. init subgroup-local online softmax state
    __align__(16) float acc_o[8] = {0.0f};
    float m_i = -FLT_MAX;
    float d_i = 0.0f;

    int phase_k[2] = {0};
    int phase_v = 0;

    const float scale_log2 = scale * 1.44269504f; // scale*log2(e)
    const int num_chunks = gridDim.x;
    const int chunk_start = chunk_id * CHUNK_SIZE;
    const int chunk_end = min(chunk_start + CHUNK_SIZE, kv_len);
    // preload Ks
    if (tid == 0) {
        mbarrier_init(mbar_k, 1);
        mbarrier_init(mbar_k + 1, 1);
        mbarrier_init(mbar_v, 1);

        mbarrier_expect_tx(mbar_k, BN * HEAD_DIM * sizeof(T));
        cp_async_bulk_tensor_3d(mbar_k, &tma_k, Ks[0], 0, kv_head_id, chunk_start);
    }
    __syncthreads();
    int read_idx = 0, write_idx = 1;

    // 5. loop over KV tiles inside this chunk
    for (int n = chunk_start; n < chunk_end; n += BN) {
        int current_bn = min(BN, chunk_end - n);

        // 5.1 TMA async load K/V
        if (tid == 0) {
            if (n + BN < chunk_end) {
                mbarrier_expect_tx(mbar_k + write_idx, BN * HEAD_DIM * sizeof(T));
                cp_async_bulk_tensor_3d(mbar_k + write_idx, &tma_k, Ks[write_idx], 0, kv_head_id, n + BN);
            }
            mbarrier_expect_tx(mbar_v, BN * HEAD_DIM * sizeof(T));
            cp_async_bulk_tensor_3d(mbar_v, &tma_v, Vs, 0, kv_head_id, n);
        }
        mbarrier_wait(mbar_k + read_idx, phase_k[read_idx]);
        phase_k[read_idx] ^= 1; // flip phase

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
                pack128 ks{FLOAT4(Ks[read_idx][row][lane_id * 8])};
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

        // next round
        __syncthreads();
        read_idx ^= 1;
        write_idx ^= 1;
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

inline int get_chunk_size(int q_head, int kv_len, int num_sms) {
    int target_blocks = num_sms * 2;

    // total_blocks = q_head * (kv_len / chunk_size)
    // chunk_size = (q_head * kv_len) / target_blocks
    int chunk = (q_head * kv_len) / target_blocks;

    if (chunk <= 64)
        return 64;
    if (chunk <= 128)
        return 128;
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
    C10_CUDA_CHECK(cudaFuncSetAttribute(                                                                               \
        NAME##_kernel<BN, CHUNK_SIZE, HEAD_DIM, 128, __nv_bfloat16>,                                                   \
        cudaFuncAttributeMaxDynamicSharedMemorySize,                                                                   \
        static_cast<int>(smem_bytes)));                                                                                \
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

#define binding_tiled_tma_func_gen(name, HEAD_DIM, kstages)                                                            \
    void name##_##HEAD_DIM(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, float scale) {    \
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
        const size_t smem_bytes =                                                                                      \
            BN * head_dim * sizeof(__nv_bfloat16) * (kstages + 1) + sizeof(mbarrier_t) * (kstages + 1);                \
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
            case 64: DISPATCH_TMA_KERNEL(name, HEAD_DIM, 64); break;                                                   \
            case 128: DISPATCH_TMA_KERNEL(name, HEAD_DIM, 128); break;                                                 \
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

binding_tiled_tma_func_gen(flash_decode_tma, 128, 1);
binding_tiled_tma_func_gen(flash_decode_tma_dbf_k, 128, 2);

#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // flash_decode_tma_128
    torch_pybinding_func(flash_decode_tma_128);
    torch_pybinding_func(flash_decode_tma_dbf_k_128);
}
