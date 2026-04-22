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

// Consumer-side arrive (flip phase) after draining smem — unused here: no producer/consumer warp specialied
// warp 0 issues loads while all warps consume; __syncthreads() is enough for our pattern.
__device__ __forceinline__ void mbarrier_arrive(mbarrier_t *mbar) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar))));
}

// Wait until the TMA copy completes. Return false on timeout so caller can diagnose instead of hanging forever.
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

// flash decoding softmax(q @ k.T*scale) @ v
template <const int BN = 64,
          const int CHUNK_SIZE = 256,
          const int HEAD_DIM = 128,
          const int THREADS_PER_BLOCK = 128,
          typename T>
__global__ void flash_decode_tma_kernel(T *q,
                                        const __grid_constant__ CUtensorMap tma_k,
                                        const __grid_constant__ CUtensorMap tma_v,
                                        float *ws_o,   // [grid_x, q_head, HEAD_DIM]
                                        float *ws_lse, // [grid_x, q_head]
                                        int kv_len,
                                        int q_head,
                                        int kv_head,
                                        float scale) {
    extern __shared__ __align__(128) uint8_t smem_buf[];
    T(*Ks)[HEAD_DIM] = reinterpret_cast<T(*)[HEAD_DIM]>(smem_buf);
    T(*Vs)[HEAD_DIM] = reinterpret_cast<T(*)[HEAD_DIM]>(smem_buf + BN * HEAD_DIM * sizeof(T));
    mbarrier_t *mbar_k = reinterpret_cast<mbarrier_t *>(smem_buf + BN * HEAD_DIM * sizeof(T) * 2);
    mbarrier_t *mbar_v = mbar_k + 1;

    // 2. coordinates
    const int tid = threadIdx.x;
    const int chunk_id = blockIdx.x;
    const int kv_head_id = blockIdx.y;
    const int group_size = q_head / kv_head;

    if (tid == 0) {
        mbarrier_init(mbar_k, 1);
        mbarrier_init(mbar_v, 1);
    }
    __syncthreads();

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int q_head_in_group = warp_id;
    const int q_head_id = kv_head_id * group_size + q_head_in_group;

    // 4. ldmatrix q
    pack64 qs{FLOAT2(q[q_head_id * HEAD_DIM + lane_id * 4])};

    // 5. Initialize output registers
    __align__(16) float acc_o[4] = {0.0f};

    // m_i and d_i track max value and softmax denominator for each of the two rows per thread
    float m_i = -FLT_MAX;
    float d_i = 0.0f;

    int phase_k = 0;
    int phase_v = 0;
    const float scale_log2 = scale * 1.44269504f; // scale*log2(e)
    const int chunk_start = chunk_id * CHUNK_SIZE;
    const int chunk_end = min(chunk_start + CHUNK_SIZE, kv_len);

    // 6. kv loop
    for (int n = chunk_start; n < chunk_end; n += BN) {
        int current_bn = min(BN, chunk_end - n);

        // --- 6.1 TMA async load KV ---
        if (tid == 0) {
            mbarrier_expect_tx(mbar_k, BN * HEAD_DIM * sizeof(T));
            mbarrier_expect_tx(mbar_v, BN * HEAD_DIM * sizeof(T));
            cp_async_bulk_tensor_4d(mbar_k, &tma_k, Ks, 0, kv_head_id, n, 0);
            cp_async_bulk_tensor_4d(mbar_v, &tma_v, Vs, 0, kv_head_id, n, 0);
        }
        __syncthreads();
        mbarrier_wait(mbar_k, phase_k);
        phase_k ^= 1; // flip phase

        // --- 6.2 Compute S = Q * K^T ---
        float acc_s[64];
#pragma unroll
        for (int row = 0; row < BN; ++row) {
            float sum = 0.0f;
            if (row < current_bn) {
                pack64 ks{FLOAT2(Ks[row][lane_id * 4])};
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    sum += static_cast<float>(qs.bf[i]) * static_cast<float>(ks.bf[i]);
                }
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum += __shfl_xor_sync(0xffffffff, sum, offset); // warp reduce
            }
            acc_s[row] = row < current_bn ? sum * scale_log2 : -FLT_MAX;
        }

        // --- 6.3 Online Softmax (Max & Correction) ---
        float m_curr = m_i;
        for (int row = 0; row < current_bn; ++row) {
            m_curr = fmaxf(m_curr, acc_s[row]);
        }
        float alpha = exp2f(m_i - m_curr);
        d_i = d_i * alpha;
        m_i = m_curr;
        // do not use lazy rescale here, not worthy
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc_o[i] *= alpha;
        }

        // --- 6.4 O = P * V ---
        mbarrier_wait(mbar_v, phase_v);
        phase_v ^= 1;
#pragma unroll 4
        for (int row = 0; row < current_bn; ++row) {
            float p = exp2f(acc_s[row] - m_i);
            d_i += p;

            pack64 vs{FLOAT2(Vs[row][lane_id * 4])};

#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc_o[i] += p * static_cast<float>(vs.bf[i]);
            }
        }
        __syncthreads();
    }

    // 7. write ws_o/ws_m/ws_d gmem
    // ws_o shape: [grid_x, q_head, head_dim]
    int out_base_idx = (chunk_id * q_head + q_head_id) * HEAD_DIM + lane_id * 4;

    pack64 out_pack;
    float inv_d = __frcp_rn(d_i);
#pragma unroll
    for (int i = 0; i < 4; i++)
        acc_o[i] *= inv_d;
    FLOAT4(ws_o[out_base_idx + 0]) = FLOAT4(acc_o[0]);

    if (lane_id == 0) {
        int scalar_idx = chunk_id * q_head + q_head_id;
        ws_lse[scalar_idx] = m_i * 0.6931471805599453f + logf(d_i);
    }
}

template <const int HEAD_DIM = 128, const int THREADS_PER_BLOCK = 32, typename T>
__global__ void flash_decode_reduce_kernel(const float *ws_o, const float *ws_lse, T *o, int num_chunks, int q_head) {
    const int q_head_id = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float s_lse;

    if (tid == 0) {
        float lse_max = -FLT_MAX;
        for (int chunk = 0; chunk < num_chunks; ++chunk) {
            lse_max = fmaxf(lse_max, ws_lse[chunk * q_head + q_head_id]);
        }

        float lse_sum = 0.0f;
        for (int chunk = 0; chunk < num_chunks; ++chunk) {
            lse_sum += expf(ws_lse[chunk * q_head + q_head_id] - lse_max);
        }
        s_lse = logf(lse_sum) + lse_max;
    }
    __syncthreads();

    const int col = tid * 4;
    if (col >= HEAD_DIM) {
        return;
    }

    float out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const int scalar_idx = chunk * q_head + q_head_id;
        const float weight = expf(ws_lse[scalar_idx] - s_lse);
        const int base_idx = scalar_idx * HEAD_DIM + col;
        out[0] += ws_o[base_idx + 0] * weight;
        out[1] += ws_o[base_idx + 1] * weight;
        out[2] += ws_o[base_idx + 2] * weight;
        out[3] += ws_o[base_idx + 3] * weight;
    }

    pack64 out_pack;
    out_pack.bf2[0] = __float22bfloat162_rn(make_float2(out[0], out[1]));
    out_pack.bf2[1] = __float22bfloat162_rn(make_float2(out[2], out[3]));
    FLOAT2(o[q_head_id * HEAD_DIM + col]) = out_pack.f2;
}

inline int get_chunk_size(int q_head, int kv_len, int num_sms) {
    int target_blocks = num_sms * 2;

    // Total_Blocks = q_head * (kv_len / chunk_size)
    // chunk_size = (q_head * kv_len) / target_blocks
    int chunk = (q_head * kv_len) / target_blocks;

    if (chunk <= 64)
        return 64;
    if (chunk <= 128)
        return 128;
    if (chunk <= 256)
        return 256;
    return 512;
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
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;

    // TMA dimensions: from fastest (0) to slowest (3)
    uint64_t globalDim[4] = {dim_d, dim_h, dim_s, dim_b};

    // globalStrides are strides for dimensions 1, 2, 3, must be in Bytes
    uint64_t globalStrides[3] = {stride_h, stride_s, stride_b};

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

#define DISPATCH_TMA_KERNEL(NAME, HEAD_DIM, CHUNK_SIZE, THREADS)                                                       \
    NAME##_kernel<BN, CHUNK_SIZE, HEAD_DIM, THREADS, __nv_bfloat16>                                                    \
        <<<blocks_per_grid, THREADS, smem_bytes, stream>>>(reinterpret_cast<__nv_bfloat16 *>(q.data_ptr()),            \
                                                           tma_k,                                                      \
                                                           tma_v,                                                      \
                                                           reinterpret_cast<float *>(ws_o.data_ptr()),                 \
                                                           reinterpret_cast<float *>(ws_lse.data_ptr()),               \
                                                           kv_len,                                                     \
                                                           q_head,                                                     \
                                                           kv_head,                                                    \
                                                           scale);

#define DISPATCH_REDUCE_KERNEL(HEAD_DIM)                                                                               \
    flash_decode_reduce_kernel<HEAD_DIM, 32, __nv_bfloat16>                                                            \
        <<<dim3(q_head), 32, 0, stream>>>(reinterpret_cast<float *>(ws_o.data_ptr()),                                  \
                                          reinterpret_cast<float *>(ws_lse.data_ptr()),                                \
                                          reinterpret_cast<__nv_bfloat16 *>(o.data_ptr()),                             \
                                          blocks_per_grid.x,                                                           \
                                          q_head);

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
        /* Build real 4D [B, S, H, D] views so TMA sees the same layout shape as flash_attn. */                        \
        torch::Tensor k4 = k.unsqueeze(0);                                                                             \
        torch::Tensor v4 = v.unsqueeze(0);                                                                             \
        int elem_bytes = k.element_size();                                                                             \
        uint64_t k_stride_h = k4.stride(2) * elem_bytes;                                                               \
        uint64_t k_stride_s = k4.stride(1) * elem_bytes;                                                               \
        uint64_t k_stride_b = k4.stride(0) * elem_bytes;                                                               \
        uint64_t v_stride_h = v4.stride(2) * elem_bytes;                                                               \
        uint64_t v_stride_s = v4.stride(1) * elem_bytes;                                                               \
        uint64_t v_stride_b = v4.stride(0) * elem_bytes;                                                               \
                                                                                                                       \
        const int BN = 64;                                                                                             \
        const int num_sms = 26;                                                                                        \
        const int chunk_size = get_chunk_size(kv_head, kv_len, num_sms);                                               \
        const int group_size = q_head / kv_head;                                                                       \
                                                                                                                       \
        CUtensorMap tma_k = create_4d_tensor_map<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16 *>(k4.data_ptr()),      \
                                                                head_dim,                                              \
                                                                kv_head,                                               \
                                                                kv_len,                                                \
                                                                1,                                                     \
                                                                k_stride_h,                                            \
                                                                k_stride_s,                                            \
                                                                k_stride_b,                                            \
                                                                head_dim,                                              \
                                                                BN);                                                   \
                                                                                                                       \
        CUtensorMap tma_v = create_4d_tensor_map<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16 *>(v4.data_ptr()),      \
                                                                head_dim,                                              \
                                                                kv_head,                                               \
                                                                kv_len,                                                \
                                                                1,                                                     \
                                                                v_stride_h,                                            \
                                                                v_stride_s,                                            \
                                                                v_stride_b,                                            \
                                                                head_dim,                                              \
                                                                BN);                                                   \
                                                                                                                       \
        TORCH_CHECK(q_head % kv_head == 0, "q_head must be divisible by kv_head");                                     \
        TORCH_CHECK(group_size == 1 || group_size == 2 || group_size == 4 || group_size == 8,                          \
                    "group_size must be 1/2/4/8");                                                                     \
        const dim3 blocks_per_grid((kv_len + chunk_size - 1) / chunk_size, kv_head);                                   \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
        const size_t smem_bytes = BN * head_dim * sizeof(__nv_bfloat16) * 2 + sizeof(mbarrier_t) * 2;                  \
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());                               \
        auto ws_lse = torch::empty({blocks_per_grid.x, q_head}, options);                                              \
        auto ws_o = torch::empty({blocks_per_grid.x, q_head, head_dim}, options);                                      \
        /* launch kernel */                                                                                            \
        switch (group_size) {                                                                                          \
            case 1:                                                                                                    \
                switch (chunk_size) {                                                                                  \
                    case 64: DISPATCH_TMA_KERNEL(name, HEAD_DIM, 64, 32); break;                                       \
                    case 128: DISPATCH_TMA_KERNEL(name, HEAD_DIM, 128, 32); break;                                     \
                    case 256: DISPATCH_TMA_KERNEL(name, HEAD_DIM, 256, 32); break;                                     \
                    case 512: DISPATCH_TMA_KERNEL(name, HEAD_DIM, 512, 32); break;                                     \
                    default: TORCH_CHECK(false, "Unsupported chunk size: ", chunk_size);                               \
                }                                                                                                      \
                break;                                                                                                 \
            case 2:                                                                                                    \
                switch (chunk_size) {                                                                                  \
                    case 64: DISPATCH_TMA_KERNEL(name, HEAD_DIM, 64, 64); break;                                       \
                    case 128: DISPATCH_TMA_KERNEL(name, HEAD_DIM, 128, 64); break;                                     \
                    case 256: DISPATCH_TMA_KERNEL(name, HEAD_DIM, 256, 64); break;                                     \
                    case 512: DISPATCH_TMA_KERNEL(name, HEAD_DIM, 512, 64); break;                                     \
                    default: TORCH_CHECK(false, "Unsupported chunk size: ", chunk_size);                               \
                }                                                                                                      \
                break;                                                                                                 \
            case 4:                                                                                                    \
                switch (chunk_size) {                                                                                  \
                    case 64: DISPATCH_TMA_KERNEL(name, HEAD_DIM, 64, 128); break;                                      \
                    case 128: DISPATCH_TMA_KERNEL(name, HEAD_DIM, 128, 128); break;                                    \
                    case 256: DISPATCH_TMA_KERNEL(name, HEAD_DIM, 256, 128); break;                                    \
                    case 512: DISPATCH_TMA_KERNEL(name, HEAD_DIM, 512, 128); break;                                    \
                    default: TORCH_CHECK(false, "Unsupported chunk size: ", chunk_size);                               \
                }                                                                                                      \
                break;                                                                                                 \
            case 8:                                                                                                    \
                switch (chunk_size) {                                                                                  \
                    case 64: DISPATCH_TMA_KERNEL(name, HEAD_DIM, 64, 256); break;                                      \
                    case 128: DISPATCH_TMA_KERNEL(name, HEAD_DIM, 128, 256); break;                                    \
                    case 256: DISPATCH_TMA_KERNEL(name, HEAD_DIM, 256, 256); break;                                    \
                    case 512: DISPATCH_TMA_KERNEL(name, HEAD_DIM, 512, 256); break;                                    \
                    default: TORCH_CHECK(false, "Unsupported chunk size: ", chunk_size);                               \
                }                                                                                                      \
                break;                                                                                                 \
            default: TORCH_CHECK(false, "Unsupported group size: ", group_size);                                       \
        }                                                                                                              \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                                \
        DISPATCH_REDUCE_KERNEL(HEAD_DIM);                                                                              \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                                \
    }

binding_tiled_tma_func_gen(flash_decode_tma, 128);

#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // flash_decode_tma_128
    torch_pybinding_func(flash_decode_tma_128);
}
