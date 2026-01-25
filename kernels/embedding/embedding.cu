#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

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

// fp32

__global__ void embedding_kernel(int *a, float *b, float *c, int emd_dim) {
    int out_offset = blockIdx.x * emd_dim;
    int in_offset = a[blockIdx.x] * emd_dim;
    int idx = threadIdx.x;
    for (int i = idx; i < emd_dim; i += blockDim.x) {
        c[out_offset + i] = b[in_offset + i];
    }
}

__global__ void embedding_fp32x4_kernel(int *a, float *b, float *c, int emd_dim) {
    int out_offset = blockIdx.x * emd_dim;
    int in_offset = a[blockIdx.x] * emd_dim;
    int idx = threadIdx.x * 4;
    for (int i = idx; i < emd_dim; i += blockDim.x * 4) {
#pragma unroll
        for (int j = 0; j < 4; j++)
            c[out_offset + i + j] = b[in_offset + i + j];
    }
}

__global__ void embedding_fp32x4_packed_kernel(int *a, float *b, float *c, int emd_dim) {
    int out_offset = blockIdx.x * emd_dim;
    int in_offset = a[blockIdx.x] * emd_dim;
    int idx = threadIdx.x * 4;
    for (int i = idx; i < emd_dim; i += blockDim.x * 4) {
        LDST128BITS(c[out_offset + i]) = LDST128BITS(b[in_offset + i]);
    }
}

// fp16
__global__ void embedding_kernel(int *a, half *b, half *c, int emd_dim) {
    int out_offset = blockIdx.x * emd_dim;
    int in_offset = a[blockIdx.x] * emd_dim;
    int idx = threadIdx.x;
    for (int i = idx; i < emd_dim; i += blockDim.x) {
        c[out_offset + i] = b[in_offset + i];
    }
}

// fp16x8
__global__ void embedding_fp16x8_kernel(int *a, half *b, half *c, int emd_dim) {
    int out_offset = blockIdx.x * emd_dim;
    int in_offset = a[blockIdx.x] * emd_dim;
    int idx = threadIdx.x * 8;
    for (int i = idx; i < emd_dim; i += blockDim.x * 8) {
#pragma unroll
        for (int j = 0; j < 8; j++)
            c[out_offset + i + j] = b[in_offset + i + j];
    }
}

// fp16x8 packed r/w
__global__ void embedding_fp16x8_packed_kernel(int *a, half *b, half *c, int emd_dim) {
    int out_offset = blockIdx.x * emd_dim;
    int in_offset = a[blockIdx.x] * emd_dim;
    int idx = threadIdx.x * 8;
    for (int i = idx; i < emd_dim; i += blockDim.x * 8) {
        LDST128BITS(c[out_offset + i]) = LDST128BITS(b[in_offset + i]);
    }
}

struct Generic {};

#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

#define binding_func_gen(name, num, element_dtype)                                                                     \
    void name(torch::Tensor a, torch::Tensor b, torch::Tensor c) {                                                     \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        CHECK_T(c);                                                                                                    \
        const int seq_len = a.size(0);                                                                                 \
        const int emd_dim = b.size(1);                                                                                 \
        const int threads_per_block = std::min(512, emd_dim) / num;                                                    \
        const int blocks_per_grid = seq_len;                                                                           \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
                                                                                                                       \
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(                                                                           \
            b.scalar_type(), #name, ([&] {                                                                             \
                using namespace std;                                                                                   \
                using cuda_t = conditional_t<is_same_v<scalar_t, at::Half>, half, scalar_t>;                           \
                using UserT = conditional_t<is_same_v<element_dtype, Generic>, cuda_t, element_dtype>;                 \
                using CastT = conditional_t<is_same_v<UserT, double>, float, UserT>;                                   \
                                                                                                                       \
                constexpr bool is_generic = is_same_v<element_dtype, Generic>;                                         \
                constexpr bool is_match = is_same_v<cuda_t, element_dtype>;                                            \
                constexpr bool is_double = is_same_v<scalar_t, double>;                                                \
                                                                                                                       \
                if constexpr (!is_double && (is_generic || is_match)) {                                                \
                    name##_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(                                  \
                        reinterpret_cast<int *>(a.data_ptr()),                                                         \
                        reinterpret_cast<CastT *>(b.data_ptr()),                                                       \
                        reinterpret_cast<CastT *>(c.data_ptr()),                                                       \
                        emd_dim);                                                                                      \
                } else {                                                                                               \
                    TORCH_CHECK(false, #name " does not support " + string(toString(b.scalar_type())));                \
                }                                                                                                      \
            }));                                                                                                       \
    }

binding_func_gen(embedding, 1, Generic);
binding_func_gen(embedding_fp32x4, 4, float);
binding_func_gen(embedding_fp32x4_packed, 4, float);
binding_func_gen(embedding_fp16x8, 8, half);
binding_func_gen(embedding_fp16x8_packed, 8, half);

// binding
#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(embedding);
    torch_pybinding_func(embedding_fp32x4);
    torch_pybinding_func(embedding_fp32x4_packed);
    torch_pybinding_func(embedding_fp16x8);
    torch_pybinding_func(embedding_fp16x8_packed);
}
