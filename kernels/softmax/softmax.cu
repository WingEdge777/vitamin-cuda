#include <ATen/cuda/CUDAContext.h>
#include <cfloat>
#include <cstdio>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

const int WARP_SIZE = 32;

struct __align__(8) MD {
    float m;
    float d;
};

// help func
template <typename T, int VEC_SIZE>
__device__ __forceinline__ float vec_max(T *ptr);

template <>
__device__ __forceinline__ float vec_max<float, 4>(float *ptr) {
    float4 v = LDST128BITS(ptr[0]);
    return fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
}

template <>
__device__ __forceinline__ float vec_max<half, 8>(half *ptr) {
    float4 v = LDST128BITS(ptr[0]);
    half2 *pack = reinterpret_cast<half2 *>(&v);
    float m = -FLT_MAX;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        float2 f2 = __half22float2(pack[i]);
        m = fmaxf(m, fmaxf(f2.x, f2.y));
    }
    return m;
}

template <typename T, int VEC_SIZE>
__device__ __forceinline__ void vec_exp_sum(T *ptr, float max_val, float &sum);

template <>
__device__ __forceinline__ void vec_exp_sum<float, 4>(float *ptr, float max_val, float &sum) {
    float4 v = LDST128BITS(ptr[0]);
    sum += __expf(v.x - max_val) + __expf(v.y - max_val) + __expf(v.z - max_val) + __expf(v.w - max_val);
}

template <>
__device__ __forceinline__ void vec_exp_sum<half, 8>(half *ptr, float max_val, float &sum) {
    float4 v = LDST128BITS(ptr[0]);
    half2 *pack = reinterpret_cast<half2 *>(&v);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        float2 f2 = __half22float2(pack[i]);
        sum += __expf(f2.x - max_val) + __expf(f2.y - max_val);
    }
}

template <typename T, int VEC_SIZE>
__device__ __forceinline__ void online_softmax_update(T *ptr, float &max_val, float &sum_val);

template <>
__device__ __forceinline__ void online_softmax_update<half, 8>(half *ptr, float &max_val, float &sum_val) {
    float4 v = LDST128BITS(ptr[0]);
    half2 *pack = reinterpret_cast<half2 *>(&v);
    float m = -FLT_MAX;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        float2 f2 = __half22float2(pack[i]);
        m = fmaxf(m, fmaxf(f2.x, f2.y));
    }
    float new_max = fmaxf(m, max_val);
    float scale = __expf(max_val - new_max);
    sum_val *= scale;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        float2 f2 = __half22float2(pack[i]);
        sum_val += __expf(f2.x - new_max) + __expf(f2.y - new_max);
    }
    max_val = new_max;
}

template <typename T, int VEC_SIZE>
__device__ __forceinline__ void vec_store_softmax(T *ptr, float max_val, float sum_val);

template <>
__device__ __forceinline__ void vec_store_softmax<float, 4>(float *ptr, float max_val, float sum_val) {
    float4 v = LDST128BITS(ptr[0]);
    float4 out;
    out.x = __expf(v.x - max_val) / sum_val;
    out.y = __expf(v.y - max_val) / sum_val;
    out.z = __expf(v.z - max_val) / sum_val;
    out.w = __expf(v.w - max_val) / sum_val;
    LDST128BITS(ptr[0]) = out;
}

template <>
__device__ __forceinline__ void vec_store_softmax<half, 8>(half *ptr, float max_val, float sum_val) {
    float4 v = LDST128BITS(ptr[0]);
    half2 *h2 = reinterpret_cast<half2 *>(&v);
    float4 out;
    half2 *out_h2 = reinterpret_cast<half2 *>(&out);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        float2 f2 = __half22float2(h2[i]);
        f2.x = __expf(f2.x - max_val) / sum_val;
        f2.y = __expf(f2.y - max_val) / sum_val;
        out_h2[i] = __float22half2_rn(f2);
    }
    LDST128BITS(ptr[0]) = out;
}

template <typename T, int VEC_SIZE>
__device__ __forceinline__ void vec_compute_softmax(T *in, T *out, float max_val, float sum_val);

template <>
__device__ __forceinline__ void vec_compute_softmax<half, 8>(half *in, half *out, float max_val, float sum_val) {
    float4 v = LDST128BITS(in[0]);
    half2 *pack = reinterpret_cast<half2 *>(&v);
    float4 res;
    half2 *res_h2 = reinterpret_cast<half2 *>(&res);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        float2 f2 = __half22float2(pack[i]);
        f2.x = __expf(f2.x - max_val) / sum_val;
        f2.y = __expf(f2.y - max_val) / sum_val;
        res_h2[i] = __float22half2_rn(f2);
    }
    LDST128BITS(out[0]) = res;
}

template <const int warp_size = WARP_SIZE>
__device__ __forceinline__ MD _warp_online_softmax_reduce(MD val) {
#pragma unroll
    for (int offset = warp_size >> 1; offset > 0; offset >>= 1) {
        float other_m = __shfl_xor_sync(0xffffffff, val.m, offset);
        float other_d = __shfl_xor_sync(0xffffffff, val.d, offset);

        float new_m = fmaxf(val.m, other_m);
        val.d = val.d * __expf(val.m - new_m) + other_d * __expf(other_m - new_m);
        val.m = new_m;
    }
    return val;
}

template <const int BLOCK_SIZE>
__device__ __forceinline__ MD block_online_softmax_reduce(MD val) {
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    val = _warp_online_softmax_reduce<WARP_SIZE>(val);

    __shared__ MD sdata[NUM_WARPS];
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (lane_id == 0)
        sdata[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = lane_id < NUM_WARPS ? sdata[lane_id] : MD{-FLT_MAX, 0.f};
        val = _warp_online_softmax_reduce<NUM_WARPS>(val);
        if (lane_id == 0)
            sdata[0] = val;
    }
    __syncthreads();
    val = sdata[0];

    return val;
}

template <const int BLOCK_SIZE>
__device__ __forceinline__ MD block_online_softmax_reduce_no_broadcast(MD val) {
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    val = _warp_online_softmax_reduce<WARP_SIZE>(val);

    __shared__ MD sdata[NUM_WARPS];
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (lane_id == 0)
        sdata[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = lane_id < NUM_WARPS ? sdata[lane_id] : MD{-FLT_MAX, 0.f};
        val = _warp_online_softmax_reduce<NUM_WARPS>(val);
    }

    return val;
}

// ============================================== kernels

template <const int BLOCK_SIZE = 256, typename T>
__global__ void softmax_kernel(T *a, T *b, int hidden_size) {
    int row_offset = blockIdx.x * hidden_size;
    int tid = threadIdx.x;

    float in[WARP_SIZE];
    MD val{-FLT_MAX, 0.f};
    int i = 0;
    for (; tid + i * BLOCK_SIZE < hidden_size; i++) {
        in[i] = static_cast<float>(a[row_offset + tid + i * BLOCK_SIZE]);
        val.m = fmaxf(val.m, in[i]);
    }
    int cnt = i;
#pragma unroll
    for (int i = 0; i < cnt; i++) {
        val.d += __expf(in[i] - val.m);
    }
    val = block_online_softmax_reduce<BLOCK_SIZE>(val);

#pragma unroll
    for (int i = 0; i < cnt; i++) {
        int offset = tid + i * BLOCK_SIZE;
        b[row_offset + offset] = static_cast<T>(__expf(in[i] - val.m) / val.d);
    }
}

template <const int BLOCK_SIZE = 256>
__global__ void softmax_fp32x4_kernel(float *a, float *b, int hidden_size) {
    int row_offset = blockIdx.x * hidden_size;
    int tid = threadIdx.x;

    float4 in[8];
    MD val{-FLT_MAX, 0.f};

    int i = 0;
#pragma unroll
    for (; (tid + i * BLOCK_SIZE) * 4 < hidden_size; i++) {
        in[i] = LDST128BITS(a[row_offset + (tid + i * BLOCK_SIZE) * 4]);
        val.m = fmaxf(val.m, vec_max<float, 4>(reinterpret_cast<float *>(&in[i])));
    }
    int cnt = i;
#pragma unroll
    for (int i = 0; i < cnt; i++) {
        vec_exp_sum<float, 4>(reinterpret_cast<float *>(&in[i]), val.m, val.d);
    }
    val = block_online_softmax_reduce<BLOCK_SIZE>(val);

#pragma unroll
    for (int i = 0; i < cnt; i++) {
        int offset = tid + i * BLOCK_SIZE;
        vec_store_softmax<float, 4>(reinterpret_cast<float *>(&in[i]), val.m, val.d);
        LDST128BITS(b[row_offset + offset * 4]) = in[i];
    }
}

template <const int BLOCK_SIZE = 256>
__global__ void softmax_fp16x8_packed_kernel(half *a, half *b, int hidden_size) {
    int row_offset = blockIdx.x * hidden_size;
    int tid = threadIdx.x;

    __align__(16) half pack[WARP_SIZE * 4];

    MD val{-FLT_MAX, 0.f};

    int i = 0;
    for (; (tid + i * BLOCK_SIZE) * 8 < hidden_size; i++) {
        LDST128BITS(pack[i * 8]) = LDST128BITS(a[row_offset + (tid + i * BLOCK_SIZE) * 8]);
        val.m = fmaxf(val.m, vec_max<half, 8>(&pack[i * 8]));
    }
    int cnt = i;

#pragma unroll
    for (int i = 0; i < cnt; i++) {
        vec_exp_sum<half, 8>(&pack[i * 8], val.m, val.d);
    }

    val = block_online_softmax_reduce<BLOCK_SIZE>(val);

#pragma unroll
    for (int i = 0; i < cnt; i++) {
        vec_compute_softmax<half, 8>(&pack[i * 8], &pack[i * 8], val.m, val.d);
        LDST128BITS(b[row_offset + (tid + i * BLOCK_SIZE) * 8]) = LDST128BITS(pack[i * 8]);
    }
}

//================================================================================================
// register + smem
template <const int BLOCK_SIZE, const int REG_VECS, const int SMEM_VECS, const int MIN_BLOCKS_PER_SM>
__global__ __launch_bounds__(BLOCK_SIZE,
                             MIN_BLOCKS_PER_SM) void softmax_onepass_kernel(half *a, half *b, int hidden_size) {
    int row_offset = blockIdx.x * hidden_size;
    int tid = threadIdx.x;

    __align__(16) half reg_pack[REG_VECS * 8];
    extern __shared__ __align__(16) half smem_dynamic[];
    half(*smem_pack)[BLOCK_SIZE * 8] = reinterpret_cast<half(*)[BLOCK_SIZE * 8]>(smem_dynamic);

    float local_m = -FLT_MAX;

#pragma unroll
    for (int i = 0; i < REG_VECS; i++) {
        int col_idx = (tid + i * BLOCK_SIZE) * 8;
        if (col_idx < hidden_size) {
            LDST128BITS(reg_pack[i * 8]) = LDST128BITS(a[row_offset + col_idx]);
            local_m = fmaxf(local_m, vec_max<half, 8>(&reg_pack[i * 8]));
        }
    }

#pragma unroll
    for (int i = 0; i < SMEM_VECS; i++) {
        int global_i = i + REG_VECS;
        int col_idx = (tid + global_i * BLOCK_SIZE) * 8;
        if (col_idx < hidden_size) {
            LDST128BITS(smem_pack[i][tid * 8]) = LDST128BITS(a[row_offset + col_idx]);
            local_m = fmaxf(local_m, vec_max<half, 8>(&smem_pack[i][tid * 8]));
        }
    }

    float local_d = 0.f;
#pragma unroll
    for (int i = 0; i < REG_VECS; i++) {
        int col_idx = (tid + i * BLOCK_SIZE) * 8;
        if (col_idx < hidden_size) {
            vec_exp_sum<half, 8>(&reg_pack[i * 8], local_m, local_d);
        }
    }
#pragma unroll
    for (int i = 0; i < SMEM_VECS; i++) {
        int global_i = i + REG_VECS;
        int col_idx = (tid + global_i * BLOCK_SIZE) * 8;
        if (col_idx < hidden_size) {
            vec_exp_sum<half, 8>(&smem_pack[i][tid * 8], local_m, local_d);
        }
    }

    MD val{local_m, local_d};
    val = block_online_softmax_reduce<BLOCK_SIZE>(val);

#pragma unroll
    for (int i = 0; i < REG_VECS; i++) {
        int col_idx = (tid + i * BLOCK_SIZE) * 8;
        if (col_idx < hidden_size) {
            vec_compute_softmax<half, 8>(&reg_pack[i * 8], &reg_pack[i * 8], val.m, val.d);
            LDST128BITS(b[row_offset + col_idx]) = LDST128BITS(reg_pack[i * 8]);
        }
    }
#pragma unroll
    for (int i = 0; i < SMEM_VECS; i++) {
        int global_i = i + REG_VECS;
        int col_idx = (tid + global_i * BLOCK_SIZE) * 8;
        if (col_idx < hidden_size) {
            vec_compute_softmax<half, 8>(&smem_pack[i][tid * 8], &smem_pack[i][tid * 8], val.m, val.d);
            LDST128BITS(b[row_offset + col_idx]) = LDST128BITS(smem_pack[i][tid * 8]);
        }
    }
}

// two pass, no cache
template <const int BLOCK_SIZE = 256>
__global__ void softmax_arbitrary_kernel(half *a, half *b, int hidden_size) {
    int row_offset = blockIdx.x * hidden_size;
    int tid = threadIdx.x;

    MD val{-FLT_MAX, 0.f};

    for (int i = 0; (tid + i * BLOCK_SIZE) * 8 < hidden_size; i++) {
        // 【修复】：强制 16 字节对齐
        __align__(16) half tmp[8];
        LDST128BITS(tmp[0]) = LDST128BITS(a[row_offset + (tid + i * BLOCK_SIZE) * 8]);
        online_softmax_update<half, 8>(tmp, val.m, val.d);
    }

    val = block_online_softmax_reduce<BLOCK_SIZE>(val);

    for (int i = 0; (tid + i * BLOCK_SIZE) * 8 < hidden_size; i++) {
        __align__(16) half tmp[8];
        LDST128BITS(tmp[0]) = LDST128BITS(a[row_offset + (tid + i * BLOCK_SIZE) * 8]);
        vec_compute_softmax<half, 8>(tmp, tmp, val.m, val.d);
        LDST128BITS(b[row_offset + (tid + i * BLOCK_SIZE) * 8]) = LDST128BITS(tmp[0]);
    }
}

// split-k pass 1
template <const int BLOCK_SIZE = 256, const int VECS_PER_THREAD = 16>
__global__ void softmax_grid_pass1(half *a, float *ws_m, float *ws_d, int hidden_size) {
    int row = blockIdx.y;
    int chunk_id = blockIdx.x;
    int tid = threadIdx.x;

    int chunk_offset = chunk_id * (BLOCK_SIZE * VECS_PER_THREAD * 8);
    int col_offset = chunk_offset + tid * 8;

    MD val{-FLT_MAX, 0.f};

    __align__(16) half cache[VECS_PER_THREAD * 8];
#pragma unroll
    for (int i = 0; i < VECS_PER_THREAD; i++) {
        int col_idx = col_offset + i * BLOCK_SIZE * 8;
        if (col_idx < hidden_size) {
            LDST128BITS(cache[i * 8]) = LDST128BITS(a[row * hidden_size + col_idx]);
            val.m = fmaxf(val.m, vec_max<half, 8>(&cache[i * 8]));
        }
    }

#pragma unroll
    for (int i = 0; i < VECS_PER_THREAD; i++) {
        int col_idx = col_offset + i * BLOCK_SIZE * 8;
        if (col_idx < hidden_size) {
            vec_exp_sum<half, 8>(&cache[i * 8], val.m, val.d);
        }
    }

    val = block_online_softmax_reduce_no_broadcast<BLOCK_SIZE>(val);

    if (tid == 0) {
        ws_m[row * gridDim.x + chunk_id] = val.m;
        ws_d[row * gridDim.x + chunk_id] = val.d;
    }
}

// splitk pass 2
template <const int BLOCK_SIZE = 256, const int VECS_PER_THREAD = 16>
__global__ void softmax_grid_pass2(half *a, half *b, float *ws_m, float *ws_d, int hidden_size) {
    int row = blockIdx.y;
    int chunk_id = blockIdx.x;
    int tid = threadIdx.x;
    int blocks_per_row = gridDim.x;

    MD global_val{-FLT_MAX, 0.f};
    for (int i = tid; i < blocks_per_row; i += BLOCK_SIZE) {
        float other_m = ws_m[row * blocks_per_row + i];
        float other_d = ws_d[row * blocks_per_row + i];
        float new_m = fmaxf(global_val.m, other_m);
        global_val.d = global_val.d * __expf(global_val.m - new_m) + other_d * __expf(other_m - new_m);
        global_val.m = new_m;
    }

    global_val = block_online_softmax_reduce<BLOCK_SIZE>(global_val);

    int chunk_offset = chunk_id * (BLOCK_SIZE * VECS_PER_THREAD * 8);
    int col_offset = chunk_offset + tid * 8;

#pragma unroll
    for (int i = 0; i < VECS_PER_THREAD; i++) {
        int col_idx = col_offset + i * BLOCK_SIZE * 8;
        if (col_idx < hidden_size) {
            // 【修复】：强制 16 字节对齐
            __align__(16) half tmp[8];
            LDST128BITS(tmp[0]) = LDST128BITS(a[row * hidden_size + col_idx]);
            vec_compute_softmax<half, 8>(tmp, tmp, global_val.m, global_val.d);
            LDST128BITS(b[row * hidden_size + col_idx]) = LDST128BITS(tmp[0]);
        }
    }
}

struct Generic {};

#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

#define binding_func_gen(name, num, element_type)                                                                      \
    void name(torch::Tensor a, torch::Tensor b) {                                                                      \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        const int bs = a.size(0);                                                                                      \
        const int lda = a.size(1);                                                                                     \
        const int threads_per_block = 256;                                                                             \
        const int blocks_per_grid = bs;                                                                                \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(                                                                           \
            a.scalar_type(), #name, ([&] {                                                                             \
                using namespace std;                                                                                   \
                using cuda_t = conditional_t<is_same_v<scalar_t, at::Half>, half, scalar_t>;                           \
                using UserT = conditional_t<is_same_v<element_type, Generic>, cuda_t, element_type>;                   \
                using CastT = conditional_t<is_same_v<UserT, double>, float, UserT>;                                   \
                constexpr bool is_generic = is_same_v<element_type, Generic>;                                          \
                constexpr bool is_match = is_same_v<cuda_t, element_type>;                                             \
                constexpr bool is_double = is_same_v<scalar_t, double>;                                                \
                if constexpr (!is_double && (is_generic || is_match)) {                                                \
                    name##_kernel<threads_per_block><<<blocks_per_grid, threads_per_block, 0, stream>>>(               \
                        reinterpret_cast<CastT *>(a.data_ptr()), reinterpret_cast<CastT *>(b.data_ptr()), lda);        \
                } else {                                                                                               \
                    TORCH_CHECK(false, #name " does not support " + string(toString(a.scalar_type())));                \
                }                                                                                                      \
            }));                                                                                                       \
    }

#define binding_single_launch_gen(name, smem_bytes, ...)                                                               \
    void name(torch::Tensor a, torch::Tensor b) {                                                                      \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        const int bs = a.size(0);                                                                                      \
        const int lda = a.size(1);                                                                                     \
        const int threads_per_block = 256;                                                                             \
        const int blocks_per_grid = bs;                                                                                \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
        if (smem_bytes > 49152) {                                                                                      \
            cudaError_t err =                                                                                          \
                cudaFuncSetAttribute(__VA_ARGS__, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);            \
            TORCH_CHECK(err == cudaSuccess, "Failed to unlock Shared Memory limit!");                                  \
        }                                                                                                              \
        __VA_ARGS__<<<blocks_per_grid, threads_per_block, smem_bytes, stream>>>(                                       \
            reinterpret_cast<half *>(a.data_ptr()), reinterpret_cast<half *>(b.data_ptr()), lda);                      \
    }

#define binding_splitk_gen(name, pass1_kernel, pass2_kernel)                                                           \
    void name(torch::Tensor a, torch::Tensor b) {                                                                      \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        const int bs = a.size(0);                                                                                      \
        const int lda = a.size(1);                                                                                     \
        const int threads_per_block = 256;                                                                             \
        constexpr int VECS_PER_THREAD = 16;                                                                            \
        int chunk_size = threads_per_block * VECS_PER_THREAD * 8;                                                      \
        int blocks_per_row = (lda + chunk_size - 1) / chunk_size;                                                      \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(a.device());                               \
        auto ws_m = torch::empty({bs, blocks_per_row}, options);                                                       \
        auto ws_d = torch::empty({bs, blocks_per_row}, options);                                                       \
        dim3 grid(blocks_per_row, bs);                                                                                 \
        pass1_kernel<threads_per_block, VECS_PER_THREAD><<<grid, threads_per_block, 0, stream>>>(                      \
            reinterpret_cast<half *>(a.data_ptr()), ws_m.data_ptr<float>(), ws_d.data_ptr<float>(), lda);              \
        pass2_kernel<threads_per_block, VECS_PER_THREAD>                                                               \
            <<<grid, threads_per_block, 0, stream>>>(reinterpret_cast<half *>(a.data_ptr()),                           \
                                                     reinterpret_cast<half *>(b.data_ptr()),                           \
                                                     ws_m.data_ptr<float>(),                                           \
                                                     ws_d.data_ptr<float>(),                                           \
                                                     lda);                                                             \
    }

binding_func_gen(softmax, 1, Generic);
binding_func_gen(softmax_fp32x4, 4, float);
binding_func_gen(softmax_fp16x8_packed, 8, half);

binding_single_launch_gen(softmax_medium, 32768, softmax_onepass_kernel<256, 8, 8, 2>);
binding_single_launch_gen(softmax_extreme, 98304, softmax_onepass_kernel<256, 32, 24, 1>);
binding_single_launch_gen(softmax_arbitrary, 0, softmax_arbitrary_kernel<256>);
binding_splitk_gen(softmax_splitk, softmax_grid_pass1, softmax_grid_pass2);

#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(softmax);
    torch_pybinding_func(softmax_fp32x4);
    torch_pybinding_func(softmax_fp16x8_packed);

    // fp16 only
    // one pass, register + smem
    torch_pybinding_func(softmax_medium);
    torch_pybinding_func(softmax_extreme);

    // two pass
    torch_pybinding_func(softmax_arbitrary);
    torch_pybinding_func(softmax_splitk);
}
