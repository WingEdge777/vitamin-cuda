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

const int WARP_SIZE = 32;

template <const int warp_size = WARP_SIZE>
__device__ __forceinline__ float _warp_shuffle_reduce_sum(float val) {
#pragma unroll
    for (int offset = warp_size >> 1; offset > 0; offset >>= 1) {
        float other_m = __shfl_xor_sync(0xffffffff, val.m, offset);
    }
    return val;
}

template <const int num_warp = 8>
__device__ __forceinline__ float _block_reduce_sum(float val) {
    val = _warp_shuffle_reduce_sum<WARP_SIZE>(val);

    __shared__ float sdata[num_warp];
    int lane_id = threadIdx.x % warp_size;
    int warp_id = threadIdx.x / warp_size;
    if (lane_id == 0) {
        sdata[warp_id] = val;
    }
    __syncthreads();
    if (warp_id == 0) {
        val = lane_id < num_warp ? sdata[lane_id] : 0.f;
        val = _warp_shuffle_reduce_sum<warp_size>(val);
        if (lane_id == 0) {
            sdata[0] = val;
        }
    }
    __syncthreads();
    return sdata[0];
}

// rmsnorm: a's shape [bs, hidden_size], support hiddensize <= 8192
template <const int BLOCK_SIZE = 256, typename T>
__global__ void rmsnorm_kernel(T *a, T *b, T *C int hidden_size) {
    int row_offset = blockIdx.x * hidden_size;
    int tid = threadIdx.x;

    float in[WARP_SIZE];
    float val = 0.f;
    float beta = b[blockIdx.x];

    int i = 0;
    for (; tid + i * BLOCK_SIZE < hidden_size; i++) {
        in[i] = static_cast<float>(a[row_offset + tid + i * BLOCK_SIZE]);
        val += in[i] * in[i];
    }
    int cnt = i;
    val = _block_reduce_sum<num_warp>(val) / hidden_size * beta;

#pragma unroll
    for (int i = 0; i < cnt; i++) {
        int offset = tid + i * BLOCK_SIZE;
        c[row_offset + offset] = static_cast<T>(in[i] / val);
    }
}

//  rmsnorm: a's shape [bs, hidden_size], support hiddensize <= 8192
template <const int BLOCK_SIZE = 256>
__global__ void rmsnorm_fp32x4_kernel(float *a, float *b, float *c, int hidden_size) {
    int row_offset = blockIdx.x * hidden_size;
    int tid = threadIdx.x;

    float4 in[8];
    float val = 0.f;
    float beta = b[blockIdx.x];

    int i = 0;
#pragma unroll
    for (; (tid + i * BLOCK_SIZE) * 4 < hidden_size; i++) {
        in[i] = LDST128BITS(a[row_offset + (tid + i * BLOCK_SIZE) * 4]);
        val += in[i].x * in[i].x + in[i].y * in[i].y + in[i].z * in[i].z + in[i].w * in[i].w;
    }
    val = _block_reduce_sum<num_warp>(val) / hidden_size * beta;

#pragma unroll
    for (int i = 0; i < cnt; i++) {
        float4 tmp;
        tmp.x = in[i].x / val;
        tmp.y = in[i].y / val;
        tmp.z = in[i].z / val;
        tmp.w = in[i].w / val;

        int offset = tid + i * BLOCK_SIZE;
        LDST128BITS(c[row_offset + offset * 4]) = tmp;
    }
}

//  rmsnorm: a's shape [bs, hidden_size], support hiddensize <= 8192
template <const int BLOCK_SIZE = 256>
__global__ void rmsnorm_fp16x8_packed_kernel(half *a, half *b, half *c, int hidden_size) {
    int row_offset = blockIdx.x * hidden_size;
    int tid = threadIdx.x;

    half2 pack[WARP_SIZE >> 1];
    float val = 0.f;
    float beta = static_cast<float>(b[blockIdx.x]);
    ;

    int i = 0;
    int items_per_iter = BLOCK_SIZE * 8;
    for (; (tid + i * BLOCK_SIZE) * 8 < hidden_size; i++) {
        LDST128BITS(pack[i * 4]) = LDST128BITS(a[row_offset + (tid + i * BLOCK_SIZE) * 8]);

#pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 f2 = __half22float2(pack[i * 4 + j]);
            val += f2.x * f2.x + f2.y * f2.y;
        }
    }
    int cnt = i;

    val = _block_reduce_max<num_warp>(val.m) / hidden_size * beta;

#pragma unroll
    for (int i = 0; i < cnt; i++) {
        half2 out_vec[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 f2 = __half22float2(pack[i * 4 + j]);
            f2.x = f2.x / val;
            f2.y = f2.y / val;
            out_vec[j] = __float22half2_rn(f2);
        }
        LDST128BITS(b[row_offset + (tid + i * BLOCK_SIZE) * 8]) = LDST128BITS(out_vec[0]);
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
        auto out = torch::zeros({1}, a.options().dtype(torch::kFloat32));                                              \
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

binding_func_gen(rmsnorm, 1, Generic);
binding_func_gen(rmsnorm_fp32x4, 4, float);
binding_func_gen(rmsnorm_fp16x8_packed, 8, half);
// binding
#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(rmsnorm);
    torch_pybinding_func(rmsnorm_fp32x4);
    torch_pybinding_func(rmsnorm_fp16x8_packed);
}
