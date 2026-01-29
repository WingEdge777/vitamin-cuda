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

__device__ __forceinline__ float _gelu(float x) { return 0.5 * x * (1.f + erff(x * 0.70710678f)); }

__device__ __forceinline__ half _gelu(half x) {
    float val = __half2float(x);
    // 使用标准 float 版本的 erff (这个所有 CUDA 版本都有)
    // 0.70710678f = 1/sqrt(2)
    val = 0.5f * val * (1.0f + erff(val * 0.70710678f));
    return __float2half(val);
}

__device__ __forceinline__ half2 _gelu(half2 x) {
    float2 vals = __half22float2(x);
    const float kAlpha = 0.70710678f;

    // 分别用 float 计算
    vals.x = 0.5f * vals.x * (1.0f + erff(vals.x * kAlpha));
    vals.y = 0.5f * vals.y * (1.0f + erff(vals.y * kAlpha));

    // 打包回 half2
    return __float22half2_rn(vals);
}

// fp32

__global__ void gelu_kernel(float *a, float *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        b[idx] = _gelu(a[idx]);
    }
}

__global__ void gelu_fp32x4_kernel(float *a, float *b, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    float4 a_val = FLOAT4(a[idx]);
    if (idx + 3 < N) {
        float4 b_val = make_float4(_gelu(a_val.x), _gelu(a_val.y), _gelu(a_val.z), _gelu(a_val.w));
        FLOAT4(b[idx]) = b_val;
    } else {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            if (idx + i < N)
                b[idx + i] = _gelu(a[idx + i]);
        }
    }
}

// fp16
__global__ void gelu_kernel(half *a, half *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        b[idx] = _gelu(a[idx]);
    }
}

// fp16x2
__global__ void gelu_fp16x2_kernel(half *a, half *b, int N) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    HALF2(b[idx]) = _gelu(HALF2(a[idx]));
}

// fp16x8
__global__ void gelu_fp16x8_kernel(half *a, half *b, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half2 a_val_0 = HALF2(a[idx]);
    half2 a_val_1 = HALF2(a[idx + 2]);
    half2 a_val_2 = HALF2(a[idx + 4]);
    half2 a_val_3 = HALF2(a[idx + 6]);

    if (idx < N) {
        HALF2(b[idx]) = _gelu(a_val_0);
    }
    if (idx + 2 < N) {
        HALF2(b[idx + 2]) = _gelu(a_val_1);
    }
    if (idx + 4 < N) {
        HALF2(b[idx + 4]) = _gelu(a_val_2);
    }
    if (idx + 6 < N) {
        HALF2(b[idx + 6]) = _gelu(a_val_3);
    }
}

// fp16x8 packed r/w
__global__ void gelu_fp16x8_packed_kernel(half *a, half *b, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx + 7 < N) {
        alignas(16) half pack_a[8];
        alignas(16) half pack_b[8];

        LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);

#pragma unroll
        for (int i = 0; i < 8; i++) {
            pack_b[i] = _gelu(pack_a[i]);
        }

        LDST128BITS(b[idx]) = LDST128BITS(pack_b[0]);
    } else {
#pragma unroll
        for (int i = 0; i < 8; i++) {
            if (idx + i < N)
                b[idx + i] = _gelu(a[idx + i]);
        }
    }
}

struct Generic {};

#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

#define binding_func_gen(name, num, element_dtype)                                                                     \
    void name(torch::Tensor a, torch::Tensor b) {                                                                      \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        const int total_elements = a.numel();                                                                          \
        const int threads_per_block = 256;                                                                             \
        const int blocks_per_grid = (total_elements / num + threads_per_block - 1) / threads_per_block;                \
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
                        reinterpret_cast<CastT *>(a.data_ptr()),                                                       \
                        reinterpret_cast<CastT *>(b.data_ptr()),                                                       \
                        total_elements);                                                                               \
                } else {                                                                                               \
                    TORCH_CHECK(false, #name " does not support " + string(toString(b.scalar_type())));                \
                }                                                                                                      \
            }));                                                                                                       \
    }

binding_func_gen(gelu, 1, Generic);
binding_func_gen(gelu_fp32x4, 4, float);
binding_func_gen(gelu_fp16x2, 2, half);
binding_func_gen(gelu_fp16x8, 8, half);
binding_func_gen(gelu_fp16x8_packed, 8, half);

// binding
#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(gelu);
    torch_pybinding_func(gelu_fp32x4);
    torch_pybinding_func(gelu_fp16x2);
    torch_pybinding_func(gelu_fp16x8);
    torch_pybinding_func(gelu_fp16x8_packed);
}
