#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/types.h>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

const float eps = 1e-6f;

const int WARP_SIZE = 32;

template <const int warp_size = WARP_SIZE>
__device__ __forceinline__ float _warp_shuffle_reduce_sum(float val) {
#pragma unroll
    for (int offset = warp_size >> 1; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

template <const int num_warp = 8>
__device__ __forceinline__ float _block_reduce_sum(float val) {
    val = _warp_shuffle_reduce_sum<WARP_SIZE>(val);

    __shared__ float sdata[num_warp];
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    if (lane_id == 0) {
        sdata[warp_id] = val;
    }
    __syncthreads();
    if (warp_id == 0) {
        val = lane_id < num_warp ? sdata[lane_id] : 0.f;
        val = _warp_shuffle_reduce_sum<num_warp>(val);
        if (lane_id == 0) {
            sdata[0] = val;
        }
    }
    __syncthreads();
    return sdata[0];
}

// rmsnorm: a's shape [bs, hidden_size], support hiddensize <= 8192
template <const int BLOCK_SIZE = 256, typename T>
__global__ void rmsnorm_kernel(T *a, T *b, T *c, int hidden_size) {
    int row_offset = blockIdx.x * hidden_size;
    int tid = threadIdx.x;

    float in[WARP_SIZE];
    float val = 0.f;

    int i = 0;
    for (; tid + i * BLOCK_SIZE < hidden_size; i++) {
        in[i] = static_cast<float>(a[row_offset + tid + i * BLOCK_SIZE]);
        val += in[i] * in[i];
        in[i] *= static_cast<float>(b[tid + i * BLOCK_SIZE]);
    }
    int cnt = i;
    val = _block_reduce_sum<BLOCK_SIZE / WARP_SIZE>(val);
    float val_inv = rsqrtf(val / hidden_size + eps);

    for (int i = 0; i < cnt; i++) {
        int offset = tid + i * BLOCK_SIZE;
        c[row_offset + offset] = static_cast<T>(in[i] * val_inv);
    }
}

//  rmsnorm: a's shape [bs, hidden_size], support hiddensize <= 8192
template <const int BLOCK_SIZE = 256>
__global__ void rmsnorm_fp32x4_kernel(float *a, float *b, float *c, int hidden_size) {
    int row_offset = blockIdx.x * hidden_size;
    int tid = threadIdx.x;

    float4 in[8];
    float val = 0.f;
    int i = 0;
    for (; (tid + i * BLOCK_SIZE) * 4 < hidden_size; i++) {
        in[i] = LDST128BITS(a[row_offset + (tid + i * BLOCK_SIZE) * 4]);
        float4 tmp = LDST128BITS(b[(tid + i * BLOCK_SIZE) * 4]);
        val += in[i].x * in[i].x + in[i].y * in[i].y + in[i].z * in[i].z + in[i].w * in[i].w;
        in[i].x = in[i].x * tmp.x;
        in[i].y = in[i].y * tmp.y;
        in[i].z = in[i].z * tmp.z;
        in[i].w = in[i].w * tmp.w;
    }
    int cnt = i;
    val = _block_reduce_sum<BLOCK_SIZE / WARP_SIZE>(val);
    float val_inv = rsqrtf(val / hidden_size + eps);

    for (int i = 0; i < cnt; i++) {
        float4 tmp;
        tmp.x = in[i].x * val_inv;
        tmp.y = in[i].y * val_inv;
        tmp.z = in[i].z * val_inv;
        tmp.w = in[i].w * val_inv;

        int offset = tid + i * BLOCK_SIZE;
        LDST128BITS(c[row_offset + offset * 4]) = tmp;
    }
}

//  rmsnorm: a's shape [bs, hidden_size], support hiddensize <= 8192
template <const int BLOCK_SIZE = 256>
__global__ void rmsnorm_fp32x4_smem_kernel(float *a, float *b, float *c, int hidden_size, int bs) {
    extern __shared__ float smem_f32[];
    int tid = threadIdx.x;
    for (int i = 0; (tid + i * BLOCK_SIZE) * 4 < hidden_size; i++) {
        LDST128BITS(smem_f32[(tid + i * BLOCK_SIZE) * 4]) = LDST128BITS(b[(tid + i * BLOCK_SIZE) * 4]);
    }
    __syncthreads();
    for (int row = blockIdx.x; row < bs; row += gridDim.x) {
        int row_offset = row * hidden_size;

        float4 in[8];
        float val = 0.f;
        int i = 0;
        for (; (tid + i * BLOCK_SIZE) * 4 < hidden_size; i++) {
            in[i] = LDST128BITS(a[row_offset + (tid + i * BLOCK_SIZE) * 4]);
            float4 tmp = LDST128BITS(smem_f32[(tid + i * BLOCK_SIZE) * 4]);
            val += in[i].x * in[i].x + in[i].y * in[i].y + in[i].z * in[i].z + in[i].w * in[i].w;
            in[i].x = in[i].x * tmp.x;
            in[i].y = in[i].y * tmp.y;
            in[i].z = in[i].z * tmp.z;
            in[i].w = in[i].w * tmp.w;
        }
        int cnt = i;
        val = _block_reduce_sum<BLOCK_SIZE / WARP_SIZE>(val);
        float val_inv = rsqrtf(val / hidden_size + eps);

        for (int i = 0; i < cnt; i++) {
            float4 tmp;
            tmp.x = in[i].x * val_inv;
            tmp.y = in[i].y * val_inv;
            tmp.z = in[i].z * val_inv;
            tmp.w = in[i].w * val_inv;

            int offset = tid + i * BLOCK_SIZE;
            LDST128BITS(c[row_offset + offset * 4]) = tmp;
        }
    }
}

//  rmsnorm: a's shape [bs, hidden_size], support hiddensize <= 8192
template <const int BLOCK_SIZE = 256>
__global__ void rmsnorm_fp16x8_packed_kernel(half *a, half *b, half *c, int hidden_size) {
    int row_offset = blockIdx.x * hidden_size;
    int tid = threadIdx.x;

    half2 pack[WARP_SIZE >> 1];
    float val = 0.f;

    int i = 0;
    for (; (tid + i * BLOCK_SIZE) * 8 < hidden_size; i++) {
        LDST128BITS(pack[i * 4]) = LDST128BITS(a[row_offset + (tid + i * BLOCK_SIZE) * 8]);
        half2 tmp[4];
        LDST128BITS(tmp[0]) = LDST128BITS(b[(tid + i * BLOCK_SIZE) * 8]);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 f2 = __half22float2(pack[i * 4 + j]);
            val += f2.x * f2.x + f2.y * f2.y;
            float2 b2 = __half22float2(tmp[j]);
            pack[i * 4 + j].x = f2.x * b2.x;
            pack[i * 4 + j].y = f2.y * b2.y;
        }
    }
    int cnt = i;

    val = _block_reduce_sum<BLOCK_SIZE / WARP_SIZE>(val);
    float val_inv = rsqrtf(val / hidden_size + eps);

    for (int i = 0; i < cnt; i++) {
        half2 out_vec[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 f2 = __half22float2(pack[i * 4 + j]);
            f2.x = f2.x * val_inv;
            f2.y = f2.y * val_inv;
            out_vec[j] = __float22half2_rn(f2);
        }
        LDST128BITS(c[row_offset + (tid + i * BLOCK_SIZE) * 8]) = LDST128BITS(out_vec[0]);
    }
}

//  rmsnorm: a's shape [bs, hidden_size], support hiddensize <= 8192
template <const int BLOCK_SIZE = 256>
__global__ void rmsnorm_fp16x8_packed_smem_kernel(half *a, half *b, half *c, int hidden_size, int bs) {
    extern __shared__ half smem_f16[];
    int tid = threadIdx.x;
    for (int i = 0; (tid + i * BLOCK_SIZE) * 8 < hidden_size; i++) {
        LDST128BITS(smem_f16[(tid + i * BLOCK_SIZE) * 8]) = LDST128BITS(b[(tid + i * BLOCK_SIZE) * 8]);
    }
    __syncthreads();
    for (int row = blockIdx.x; row < bs; row += gridDim.x) {
        int row_offset = row * hidden_size;

        half2 pack[WARP_SIZE >> 1];
        float val = 0.f;

        int i = 0;
        for (; (tid + i * BLOCK_SIZE) * 8 < hidden_size; i++) {
            LDST128BITS(pack[i * 4]) = LDST128BITS(a[row_offset + (tid + i * BLOCK_SIZE) * 8]);
            half2 tmp[4];
            LDST128BITS(tmp[0]) = LDST128BITS(smem_f16[(tid + i * BLOCK_SIZE) * 8]);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float2 f2 = __half22float2(pack[i * 4 + j]);
                val += f2.x * f2.x + f2.y * f2.y;
                float2 b2 = __half22float2(tmp[j]);
                pack[i * 4 + j].x = f2.x * b2.x;
                pack[i * 4 + j].y = f2.y * b2.y;
            }
        }
        int cnt = i;

        val = _block_reduce_sum<BLOCK_SIZE / WARP_SIZE>(val);
        float val_inv = rsqrtf(val / hidden_size + eps);

        for (int i = 0; i < cnt; i++) {
            half2 out_vec[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float2 f2 = __half22float2(pack[i * 4 + j]);
                f2.x = f2.x * val_inv;
                f2.y = f2.y * val_inv;
                out_vec[j] = __float22half2_rn(f2);
            }
            LDST128BITS(c[row_offset + (tid + i * BLOCK_SIZE) * 8]) = LDST128BITS(out_vec[0]);
        }
    }
}

struct Generic {};

#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

#define binding_func_gen(name, num, element_type)                                                                      \
    torch::Tensor name(torch::Tensor a, torch::Tensor b) {                                                             \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        const int bs = a.size(0);                                                                                      \
        const int lda = a.size(1);                                                                                     \
        const int threads_per_block = 256;                                                                             \
        const int blocks_per_grid = bs;                                                                                \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
        auto out = torch::empty_like(a);                                                                               \
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(                                                                           \
            a.scalar_type(), #name, ([&] {                                                                             \
                using namespace std;                                                                                   \
                using cuda_t = conditional_t<is_same_v<scalar_t, at::Half>, half, scalar_t>;                           \
                using UserT = conditional_t<is_same_v<element_type, Generic>, cuda_t, element_type>;                   \
                using CastT = conditional_t<is_same_v<UserT, double>, float, UserT>;                                   \
                                                                                                                       \
                constexpr bool is_generic = is_same_v<element_type, Generic>;                                          \
                constexpr bool is_match = is_same_v<cuda_t, element_type>;                                             \
                constexpr bool is_double = is_same_v<scalar_t, double>;                                                \
                                                                                                                       \
                if constexpr (!is_double && (is_generic || is_match)) {                                                \
                    name##_kernel<threads_per_block>                                                                   \
                        <<<blocks_per_grid, threads_per_block, 0, stream>>>(reinterpret_cast<CastT *>(a.data_ptr()),   \
                                                                            reinterpret_cast<CastT *>(b.data_ptr()),   \
                                                                            reinterpret_cast<CastT *>(out.data_ptr()), \
                                                                            lda);                                      \
                } else {                                                                                               \
                    TORCH_CHECK(false, #name " does not support " + string(toString(a.scalar_type())));                \
                }                                                                                                      \
            }));                                                                                                       \
        return out;                                                                                                    \
    }

#define binding_smem_func_gen(name, rows_per_block, element_type)                                                      \
    torch::Tensor name(torch::Tensor a, torch::Tensor b) {                                                             \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        const int bs = a.size(0);                                                                                      \
        const int lda = a.size(1);                                                                                     \
        const int threads_per_block = 256;                                                                             \
        const int blocks_per_grid = (bs + rows_per_block - 1) / rows_per_block;                                        \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
        auto out = torch::empty_like(a);                                                                               \
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(                                                                           \
            a.scalar_type(), #name, ([&] {                                                                             \
                using namespace std;                                                                                   \
                using cuda_t = conditional_t<is_same_v<scalar_t, at::Half>, half, scalar_t>;                           \
                using UserT = conditional_t<is_same_v<element_type, Generic>, cuda_t, element_type>;                   \
                using CastT = conditional_t<is_same_v<UserT, double>, float, UserT>;                                   \
                                                                                                                       \
                constexpr bool is_generic = is_same_v<element_type, Generic>;                                          \
                constexpr bool is_match = is_same_v<cuda_t, element_type>;                                             \
                constexpr bool is_double = is_same_v<scalar_t, double>;                                                \
                                                                                                                       \
                if constexpr (!is_double && (is_generic || is_match)) {                                                \
                    name##_kernel<threads_per_block>                                                                   \
                        <<<blocks_per_grid, threads_per_block, lda * sizeof(CastT), stream>>>(                         \
                            reinterpret_cast<CastT *>(a.data_ptr()),                                                   \
                            reinterpret_cast<CastT *>(b.data_ptr()),                                                   \
                            reinterpret_cast<CastT *>(out.data_ptr()),                                                 \
                            lda,                                                                                       \
                            bs);                                                                                       \
                } else {                                                                                               \
                    TORCH_CHECK(false, #name " does not support " + string(toString(a.scalar_type())));                \
                }                                                                                                      \
            }));                                                                                                       \
        return out;                                                                                                    \
    }

binding_func_gen(rmsnorm, 1, Generic);
binding_func_gen(rmsnorm_fp32x4, 4, float);
binding_smem_func_gen(rmsnorm_fp32x4_smem, 16, float);
binding_func_gen(rmsnorm_fp16x8_packed, 8, half);
binding_smem_func_gen(rmsnorm_fp16x8_packed_smem, 16, half);
// binding
#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(rmsnorm);
    torch_pybinding_func(rmsnorm_fp32x4);
    torch_pybinding_func(rmsnorm_fp32x4_smem);
    torch_pybinding_func(rmsnorm_fp16x8_packed);
    torch_pybinding_func(rmsnorm_fp16x8_packed_smem);
}
