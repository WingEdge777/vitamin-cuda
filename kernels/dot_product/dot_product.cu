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

template <const int warp_size = WARP_SIZE, typename T>
__device__ __forceinline__ T _warp_shuffle_reduce(T x) {
#pragma unroll
    for (int mask = warp_size >> 1; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask);
    }
    return x;
}

// fp32
template <const int block_size = 256, typename T>
__global__ void dot_product_kernel(T *a, T *b, float *c, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;

    for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
        sum += static_cast<float>(a[i] * b[i]);
    }

    const int num_warp = (block_size + WARP_SIZE - 1) / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    __shared__ float smem[num_warp];

    sum = _warp_shuffle_reduce<WARP_SIZE>(sum);
    if (lane_id == 0) {
        smem[warp_id] = sum;
    }
    __syncthreads();

    sum = lane_id < num_warp ? smem[lane_id] : 0.f;

    if (warp_id == 0) {
        sum = _warp_shuffle_reduce(sum);
        if (tid == 0) {
            atomicAdd(c, sum);
        }
    }
}

template <const int block_size = 256>
__global__ void dot_product_fp32x4_kernel(float *a, float *b, float *c, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;
    float4 a_val, b_val;
    for (int i = idx * 4; i < N; i += gridDim.x * blockDim.x * 4) {
        a_val = FLOAT4(a[i]);
        b_val = FLOAT4(b[i]);
        sum += a_val.x * b_val.x + a_val.y * b_val.y + a_val.z * b_val.z + a_val.w * b_val.w;
    }

    const int num_warp = (block_size + WARP_SIZE - 1) / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    __shared__ float smem[num_warp];

    sum = _warp_shuffle_reduce<WARP_SIZE>(sum);
    if (lane_id == 0) {
        smem[warp_id] = sum;
    }
    __syncthreads();

    sum = lane_id < num_warp ? smem[lane_id] : 0.f;

    if (warp_id == 0) {
        sum = _warp_shuffle_reduce(sum);
        if (tid == 0) {
            atomicAdd(c, sum);
        }
    }
}

template <const int block_size = 256>
__global__ void dot_product_fp16x2_kernel(half *a, half *b, float *c, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;
    for (int i = idx * 2; i < N; i += gridDim.x * blockDim.x * 2) {
        half2 a_val = HALF2(a[i]);
        half2 b_val = HALF2(b[i]);
        float2 t = __half22float2(__hmul2(a_val, b_val));
        sum += t.x + t.y;
    }

    const int num_warp = (block_size + WARP_SIZE - 1) / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    __shared__ float smem[num_warp];

    sum = _warp_shuffle_reduce<WARP_SIZE>(sum);
    if (lane_id == 0) {
        smem[warp_id] = sum;
    }
    __syncthreads();

    sum = lane_id < num_warp ? smem[lane_id] : 0.f;

    if (warp_id == 0) {
        sum = _warp_shuffle_reduce(sum);
        if (tid == 0) {
            atomicAdd(c, sum);
        }
    }
}

template <const int block_size = 256>
__global__ void dot_product_fp16x8_packed_kernel(half *a, half *b, float *c, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;
    alignas(16) half2 p[4], v[4];
    for (int i = idx * 8; i < N; i += gridDim.x * blockDim.x * 8) {
        LDST128BITS(p[0]) = LDST128BITS(a[i]);
        LDST128BITS(v[0]) = LDST128BITS(b[i]);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 t = __half22float2(__hmul2(p[j], v[j]));
            sum += t.x + t.y;
        }
    }

    const int num_warp = (block_size + WARP_SIZE - 1) / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    __shared__ float smem[num_warp];

    sum = _warp_shuffle_reduce<WARP_SIZE>(sum);
    if (lane_id == 0) {
        smem[warp_id] = sum;
    }
    __syncthreads();

    sum = lane_id < num_warp ? smem[lane_id] : 0.f;

    if (warp_id == 0) {
        sum = _warp_shuffle_reduce(sum);
        if (tid == 0) {
            atomicAdd(c, sum);
        }
    }
}

struct Generic {};

#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

#define binding_func_gen(name, num, element_dtype)                                                                     \
    torch::Tensor name(torch::Tensor a, torch::Tensor b) {                                                             \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        const int total_elements = a.numel();                                                                          \
        const int threads_per_block = 256;                                                                             \
        const int blocks_per_grid =                                                                                    \
            std::min((total_elements / num + threads_per_block - 1) / threads_per_block, 1024);                        \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
        auto out = torch::zeros({1}, a.options().dtype(torch::kFloat32));                                              \
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(                                                                           \
            a.scalar_type(), #name, ([&] {                                                                             \
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
                        reinterpret_cast<float *>(out.data_ptr()),                                                     \
                        total_elements);                                                                               \
                } else {                                                                                               \
                    TORCH_CHECK(false, #name " does not support " + string(toString(a.scalar_type())));                \
                }                                                                                                      \
            }));                                                                                                       \
        return out.to(a.scalar_type());                                                                                \
    }

binding_func_gen(dot_product, 1, Generic);
binding_func_gen(dot_product_fp32x4, 4, float);
binding_func_gen(dot_product_fp16x2, 2, half);
binding_func_gen(dot_product_fp16x8_packed, 8, half);

// binding
#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(dot_product);
    torch_pybinding_func(dot_product_fp32x4);
    torch_pybinding_func(dot_product_fp16x2);
    torch_pybinding_func(dot_product_fp16x8_packed);
}
