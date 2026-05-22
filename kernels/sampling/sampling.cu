#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/types.h>

#include "../common/pack.cuh"

// stable insertion sort
template <const int TOP_K>
__device__ __forceinline__ bool insert_sorted(float score[TOP_K], int token_id[TOP_K], float new_val, int new_id) {
    if (new_val <= score[TOP_K - 1])
        return false;

#pragma unroll
    for (int i = TOP_K - 1; i > 0; i--) {
        bool bigger_than_curr = (new_val > score[i]);
        bool bigger_than_prev = (new_val > score[i - 1]);

        score[i] = bigger_than_curr ? (bigger_than_prev ? score[i - 1] : new_val) : score[i];
        token_id[i] = bigger_than_curr ? (bigger_than_prev ? token_id[i - 1] : new_id) : token_id[i];
    }

    bool bigger_than_0 = (new_val > score[0]);
    score[0] = bigger_than_0 ? new_val : score[0];
    token_id[0] = bigger_than_0 ? new_id : token_id[0];
    return true;
}

template <const int TOP_K = 20, const int CHUNK_SIZE = 2048, const int BLOCK_SIZE = 256, typename T>
__global__ void sampling_topk_topp_batched_kernel(
    T *logits, int *output_ids, float top_p, int64_t seed, int64_t offset, int vocab_size) {
    const int batch_id = blockIdx.x;
    const int tid = threadIdx.x * 8;

    T *row_ptr = logits + batch_id * vocab_size;
    // step 1: maintain sorted local score/token_id array
    float score[TOP_K];
    int token_id[TOP_K];
#pragma unroll
    for (int i = 0; i < TOP_K; i++) {
        score[i] = -FLT_MAX;
        token_id[i] = -1;
    }
    for (int idx = tid; idx < vocab_size; idx += CHUNK_SIZE) {
        pack128 tmp;
        tmp.f4 = FLOAT4(row_ptr[idx]);
        for (int x = 0; x < 8; x++) {
            float val = static_cast<float>(tmp.h[x]);
            int v_idx = idx + x;
            if (val > score[TOP_K - 1]) {
                insert_sorted<TOP_K>(score, token_id, val, v_idx);
            }
        }
    }

    // step 2: warp reduce merge local array
    int lane_id = threadIdx.x % 32;
#pragma unroll
    for (int src_line = 16; src_line > 0; src_line /= 2) {

#pragma unroll
        for (int j = 0; j < TOP_K; ++j) {
            float other_val = __shfl_down_sync(0xffffffff, score[j], src_line);
            int other_id = __shfl_down_sync(0xffffffff, token_id[j], src_line);

            if (lane_id < src_line) {
                insert_sorted<TOP_K>(score, token_id, other_val, other_id);
            }
        }
    }
    // step 3: final reduce and sampling
    const int num_warps = BLOCK_SIZE / WARP_SIZE;
    __shared__ float smem_warp_score[num_warps][TOP_K];
    __shared__ int smem_warp_id[num_warps][TOP_K];

    int warp_id = threadIdx.x / WARP_SIZE;

    if (lane_id == 0) {
#pragma unroll
        for (int i = 0; i < TOP_K; i++) {
            smem_warp_score[warp_id][i] = score[i];
            smem_warp_id[warp_id][i] = token_id[i];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // curand
        curandStatePhilox4_32_10_t state;
        curand_init(seed, blockIdx.x, offset, &state);
        float u = curand_uniform(&state); // create a float from (0, 1]
        float random_val = 1.f - u;

        // block reduce
        for (int w = 1; w < num_warps; w++) {
#pragma unroll
            for (int j = 0; j < TOP_K; ++j) {
                float other_val = smem_warp_score[w][j];
                int other_id = smem_warp_id[w][j];
                if (!insert_sorted<TOP_K>(score, token_id, other_val, other_id)) {
                    break;
                }
            }
        }

        // softmax
        float max_val = score[0];
        float sum_prob = 0.0f;
        float probs[TOP_K];
#pragma unroll
        for (int i = 0; i < TOP_K; i++) {
            probs[i] = expf(score[i] - max_val);
            sum_prob += probs[i];
        }

        // top_p
        float cumsum = 0.0f;
        float trunc_sum = 0.0f;
        int last_idx = TOP_K - 1;

        for (int i = 0; i < TOP_K; i++) {
            float p = probs[i] / sum_prob;
            cumsum += p;
            if (cumsum >= top_p) {
                last_idx = i;
                trunc_sum = cumsum;
                break;
            }
        }
        if (trunc_sum == 0.0f)
            trunc_sum = cumsum;

        // sampling
        float r = random_val * trunc_sum;
        float cdf = 0.0f;
        int final_id = token_id[last_idx];

        for (int i = 0; i <= last_idx; i++) {
            cdf += probs[i] / sum_prob;
            if (cdf >= r) {
                final_id = token_id[i];
                break;
            }
        }

        // final token id
        output_ids[blockIdx.x] = final_id;
    }
}

// split-k pass 1: partial topk per split
template <const int TOP_K = 20, const int CHUNK_SIZE = 2048, const int BLOCK_SIZE = 256, typename T>
__global__ void sampling_topk_topp_split_k_pass1_kernel(T *logits,
                                                         int vocab_size,
                                                         float *ws_score,
                                                         int *ws_id) {
    const int split_id = blockIdx.x;
    const int batch_id = blockIdx.y;
    const int num_splits = gridDim.x;

    const int vocab_per_split = ((vocab_size + num_splits * 8 - 1) / (num_splits * 8)) * 8;
    const int vocab_start = split_id * vocab_per_split;
    const int vocab_end = min(vocab_start + vocab_per_split, vocab_size);

    T *row_ptr = logits + batch_id * vocab_size;
    int tid = threadIdx.x * 8;
    // step 1: maintain sorted local score/token_id array
    float score[TOP_K];
    int token_id[TOP_K];
#pragma unroll
    for (int i = 0; i < TOP_K; i++) {
        score[i] = -FLT_MAX;
        token_id[i] = -1;
    }
    for (int idx = vocab_start + tid; idx < vocab_end; idx += CHUNK_SIZE) {
        pack128 tmp;
        tmp.f4 = FLOAT4(row_ptr[idx]);
        for (int x = 0; x < 8; x++) {
            float val = static_cast<float>(tmp.h[x]);
            int v_idx = idx + x;
            if (val > score[TOP_K - 1]) {
                insert_sorted<TOP_K>(score, token_id, val, v_idx);
            }
        }
    }
    // step 2: warp reduce merge local array
    int lane_id = threadIdx.x % 32;
#pragma unroll
    for (int src_line = 16; src_line > 0; src_line /= 2) {
#pragma unroll
        for (int j = 0; j < TOP_K; ++j) {
            float other_val = __shfl_down_sync(0xffffffff, score[j], src_line);
            int other_id = __shfl_down_sync(0xffffffff, token_id[j], src_line);
            if (lane_id < src_line) {
                insert_sorted<TOP_K>(score, token_id, other_val, other_id);
            }
        }
    }
    // step 3: block reduce merge and write to gmem
    const int num_warps = BLOCK_SIZE / WARP_SIZE;
    __shared__ float smem_warp_score[num_warps][TOP_K];
    __shared__ int smem_warp_id[num_warps][TOP_K];
    int warp_id = threadIdx.x / WARP_SIZE;

    if (lane_id == 0) {
#pragma unroll
        for (int i = 0; i < TOP_K; i++) {
            smem_warp_score[warp_id][i] = score[i];
            smem_warp_id[warp_id][i] = token_id[i];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // block reduce
        for (int w = 1; w < num_warps; w++) {
#pragma unroll
            for (int j = 0; j < TOP_K; ++j) {
                float other_val = smem_warp_score[w][j];
                int other_id = smem_warp_id[w][j];
                if (!insert_sorted<TOP_K>(score, token_id, other_val, other_id)) {
                    break;
                }
            }
        }
        int ws_offset = (batch_id * num_splits + split_id) * TOP_K;
        for (int i = 0; i < TOP_K; i++) {
            ws_score[ws_offset + i] = score[i];
            ws_id[ws_offset + i] = token_id[i];
        }
    }
}

// split-k pass 2: merge partial topk and sampling
template <const int TOP_K = 20, const int BLOCK_SIZE=256>
__global__ void sampling_topk_topp_split_k_pass2_kernel(int *output_ids,
                                                         float top_p,
                                                         int64_t seed,
                                                         int64_t offset,
                                                         int num_splits,
                                                         float *ws_score,
                                                         int *ws_id) {
    const int batch_id = blockIdx.x;
    const int total = num_splits * TOP_K;
    const int ws_base = batch_id * total;

    float score[TOP_K];
    int token_id[TOP_K];
#pragma unroll
    for (int i = 0; i < TOP_K; i++) {
        score[i] = -FLT_MAX;
        token_id[i] = -1;
    }

    for (int idx = threadIdx.x; idx < total; idx += BLOCK_SIZE) {
        insert_sorted<TOP_K>(score, token_id, ws_score[ws_base + idx], ws_id[ws_base + idx]);
    }

    // warp reduce merge
    int lane_id = threadIdx.x % 32;
#pragma unroll
    for (int src_line = 16; src_line > 0; src_line /= 2) {
#pragma unroll
        for (int j = 0; j < TOP_K; ++j) {
            float other_val = __shfl_down_sync(0xffffffff, score[j], src_line);
            int other_id = __shfl_down_sync(0xffffffff, token_id[j], src_line);
            if (lane_id < src_line) {
                insert_sorted<TOP_K>(score, token_id, other_val, other_id);
            }
        }
    }

    const int num_warps = BLOCK_SIZE / WARP_SIZE;
    __shared__ float smem_warp_score[num_warps][TOP_K];
    __shared__ int smem_warp_id[num_warps][TOP_K];
    int warp_id = threadIdx.x / WARP_SIZE;

    if (lane_id == 0) {
#pragma unroll
        for (int i = 0; i < TOP_K; i++) {
            smem_warp_score[warp_id][i] = score[i];
            smem_warp_id[warp_id][i] = token_id[i];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {

        // block reduce
        for (int w = 1; w < num_warps; w++) {
#pragma unroll
            for (int j = 0; j < TOP_K; ++j) {
                float other_val = smem_warp_score[w][j];
                int other_id = smem_warp_id[w][j];
                if (!insert_sorted<TOP_K>(score, token_id, other_val, other_id)) {
                    break;
                }
            }
        }

        // curand
        curandStatePhilox4_32_10_t state;
        curand_init(seed, batch_id, offset, &state);
        float u = curand_uniform(&state);
        float random_val = 1.f - u;

        // softmax
        float max_val = score[0];
        float sum_prob = 0.0f;
        float probs[TOP_K];
#pragma unroll
        for (int i = 0; i < TOP_K; i++) {
            probs[i] = expf(score[i] - max_val);
            sum_prob += probs[i];
        }

        // top_p
        float cumsum = 0.0f;
        float trunc_sum = 0.0f;
        int last_idx = TOP_K - 1;
        for (int i = 0; i < TOP_K; i++) {
            float p = probs[i] / sum_prob;
            cumsum += p;
            if (cumsum >= top_p) {
                last_idx = i;
                trunc_sum = cumsum;
                break;
            }
        }
        if (trunc_sum == 0.0f)
            trunc_sum = 1.0f;

        // sampling
        float r = random_val * trunc_sum;
        float cdf = 0.0f;
        int final_id = token_id[last_idx];
        for (int i = 0; i <= last_idx; i++) {
            cdf += probs[i] / sum_prob;
            if (cdf >= r) {
                final_id = token_id[i];
                break;
            }
        }

        // final token id
        output_ids[batch_id] = final_id;
    }
}

#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

#define DISPATCH_TOPK_KERNEL(kernel_name, K_VAL)                                                                       \
    if (logits.dtype() == torch::kHalf) {                                                                              \
        kernel_name<K_VAL, chunk_size>                                                                                 \
            <<<blocks_per_grid, threads_per_block, 0, stream>>>(reinterpret_cast<__half *>(logits.data_ptr()),         \
                                                                reinterpret_cast<int *>(res.data_ptr()),               \
                                                                top_p,                                                 \
                                                                seed,                                                  \
                                                                offset,                                                \
                                                                vocab_size);                                           \
    } else if (logits.dtype() == torch::kBFloat16) {                                                                   \
        kernel_name<K_VAL, chunk_size>                                                                                 \
            <<<blocks_per_grid, threads_per_block, 0, stream>>>(reinterpret_cast<__nv_bfloat16 *>(logits.data_ptr()),  \
                                                                reinterpret_cast<int *>(res.data_ptr()),               \
                                                                top_p,                                                 \
                                                                seed,                                                  \
                                                                offset,                                                \
                                                                vocab_size);                                           \
    }

#define binding_tiled_func_gen(name)                                                                                   \
    torch::Tensor name(torch::Tensor logits, int top_k, float top_p, int64_t seed, int64_t offset) {                   \
        CHECK_T(logits);                                                                                               \
        const int bs = logits.size(0);                                                                                 \
        const int vocab_size = logits.size(1);                                                                         \
        constexpr int threads_per_block = 256;                                                                         \
        constexpr int chunk_size = threads_per_block * 8;                                                              \
        const dim3 blocks_per_grid(bs);                                                                                \
        auto options = torch::TensorOptions().dtype(torch::kInt32).device(logits.device());                            \
        auto res = torch::empty({bs}, options);                                                                        \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
                                                                                                                       \
        switch (top_k) {                                                                                               \
            case 1: DISPATCH_TOPK_KERNEL(name##_kernel, 1); break;                                                     \
            case 2: DISPATCH_TOPK_KERNEL(name##_kernel, 2); break;                                                     \
            case 4: DISPATCH_TOPK_KERNEL(name##_kernel, 4); break;                                                     \
            case 8: DISPATCH_TOPK_KERNEL(name##_kernel, 8); break;                                                     \
            case 16: DISPATCH_TOPK_KERNEL(name##_kernel, 16); break;                                                   \
            case 20: DISPATCH_TOPK_KERNEL(name##_kernel, 20); break;                                                   \
            default: TORCH_CHECK(false, "Unsupported top_k! Only powers of 2 up to 32 are supported."); break;         \
        }                                                                                                              \
        return res;                                                                                                    \
    }

#define DISPATCH_TOPK_SPLITK_KERNEL(kernel_name, K_VAL)                                                                \
    if (logits.dtype() == torch::kHalf) {                                                                              \
        kernel_name<K_VAL, chunk_size><<<blocks_p1, threads_per_block, 0, stream>>>(                                   \
            reinterpret_cast<__half *>(logits.data_ptr()),                                                             \
            vocab_size,                                                                                                 \
            ws_score.data_ptr<float>(),                                                                                 \
            ws_id.data_ptr<int>());                                                                                     \
        sampling_topk_topp_split_k_pass2_kernel<K_VAL><<<blocks_p2, threads_per_block, 0, stream>>>(                                   \
            reinterpret_cast<int *>(res.data_ptr()),                                                                   \
            top_p,                                                                                                      \
            seed,                                                                                                       \
            offset,                                                                                                     \
            num_splits,                                                                                                 \
            ws_score.data_ptr<float>(),                                                                                 \
            ws_id.data_ptr<int>());                                                                                     \
    } else if (logits.dtype() == torch::kBFloat16) {                                                                   \
        kernel_name<K_VAL, chunk_size><<<blocks_p1, threads_per_block, 0, stream>>>(                                   \
            reinterpret_cast<__nv_bfloat16 *>(logits.data_ptr()),                                                      \
            vocab_size,                                                                                                 \
            ws_score.data_ptr<float>(),                                                                                 \
            ws_id.data_ptr<int>());                                                                                     \
        sampling_topk_topp_split_k_pass2_kernel<K_VAL><<<blocks_p2, threads_per_block, 0, stream>>>(                                   \
            reinterpret_cast<int *>(res.data_ptr()),                                                                   \
            top_p,                                                                                                      \
            seed,                                                                                                       \
            offset,                                                                                                     \
            num_splits,                                                                                                 \
            ws_score.data_ptr<float>(),                                                                                 \
            ws_id.data_ptr<int>());                                                                                     \
    }

torch::Tensor sampling_topk_topp_split_k(torch::Tensor logits, int top_k, float top_p, int64_t seed, int64_t offset) {
    CHECK_T(logits);
    const int bs = logits.size(0);
    const int vocab_size = logits.size(1);
    constexpr int threads_per_block = 256;
    constexpr int chunk_size = threads_per_block * 8;
    constexpr int num_splits = 26;
    const dim3 blocks_p1(num_splits, bs);
    const int blocks_p2 = bs;
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(logits.device());
    auto res = torch::empty({bs}, options);
    auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(logits.device());
    auto ws_score = torch::empty({bs * num_splits * top_k}, float_opts);
    auto ws_id = torch::empty({bs * num_splits * top_k}, options);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    switch (top_k) {
        case 1: DISPATCH_TOPK_SPLITK_KERNEL(sampling_topk_topp_split_k_pass1_kernel, 1); break;
        case 2: DISPATCH_TOPK_SPLITK_KERNEL(sampling_topk_topp_split_k_pass1_kernel, 2); break;
        case 4: DISPATCH_TOPK_SPLITK_KERNEL(sampling_topk_topp_split_k_pass1_kernel, 4); break;
        case 8: DISPATCH_TOPK_SPLITK_KERNEL(sampling_topk_topp_split_k_pass1_kernel, 8); break;
        case 16: DISPATCH_TOPK_SPLITK_KERNEL(sampling_topk_topp_split_k_pass1_kernel, 16); break;
        case 20: DISPATCH_TOPK_SPLITK_KERNEL(sampling_topk_topp_split_k_pass1_kernel, 20); break;
        default: TORCH_CHECK(false, "Unsupported top_k! Only powers of 2 up to 32 are supported."); break;
    }
    return res;
}

binding_tiled_func_gen(sampling_topk_topp_batched);

#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(sampling_topk_topp_batched);
    torch_pybinding_func(sampling_topk_topp_split_k);
}
