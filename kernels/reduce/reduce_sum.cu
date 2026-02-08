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

template <const int block_size = 256, typename T>
__global__ void reduce_sum_kernel(T *a, float *b, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;

    for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
        sum += static_cast<float>(a[i]);
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

    if (warp_id == 0) {
        sum = lane_id < num_warp ? smem[lane_id] : 0.f;
        sum = _warp_shuffle_reduce<num_warp>(sum);
        if (tid == 0) {
            atomicAdd(b, sum);
        }
    }
}

template <const int block_size = 256>
__global__ void reduce_sum_fp32x4_kernel(float *a, float *b, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;
    float4 a_val;
    for (int i = idx * 4; i < N; i += gridDim.x * blockDim.x * 4) {
        a_val = FLOAT4(a[i]);
        sum += a_val.x + a_val.y + a_val.z + a_val.w;
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

    if (warp_id == 0) {
        sum = lane_id < num_warp ? smem[lane_id] : 0.f;
        sum = _warp_shuffle_reduce<num_warp>(sum);
        if (tid == 0) {
            atomicAdd(b, sum);
        }
    }
}

template <const int block_size = 256>
__global__ void reduce_sum_fp16x2_kernel(half *a, float *b, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;
    for (int i = idx * 2; i < N; i += gridDim.x * blockDim.x * 2) {
        float2 p = __half22float2(HALF2(a[i]));
        sum += p.x + p.y;
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

    if (warp_id == 0) {
        sum = lane_id < num_warp ? smem[lane_id] : 0.f;
        sum = _warp_shuffle_reduce<num_warp>(sum);
        if (tid == 0) {
            atomicAdd(b, sum);
        }
    }
}

template <const int block_size = 256>
__global__ void reduce_sum_fp16x8_packed_kernel(half *a, float *b, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;
    alignas(16) half2 p[4];
    for (int i = idx * 8; i < N; i += gridDim.x * blockDim.x * 8) {
        LDST128BITS(p[0]) = LDST128BITS(a[i]);
#pragma unroll
        for (int t = 0; t < 4; t++) {
            float2 f = __half22float2(p[t]);
            sum += f.x + f.y;
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

    if (warp_id == 0) {
        sum = lane_id < num_warp ? smem[lane_id] : 0.f;
        sum = _warp_shuffle_reduce<num_warp>(sum);
        if (tid == 0) {
            atomicAdd(b, sum);
        }
    }
}

template <const int block_size = 256>
__global__ void reduce_sum_i8_kernel(int8_t *a, int32_t *b, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int32_t sum = 0;
    for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
        sum += static_cast<int32_t>(a[i]);
    }

    const int num_warp = (block_size + WARP_SIZE - 1) / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    __shared__ int smem[num_warp];

    sum = _warp_shuffle_reduce<WARP_SIZE>(sum);
    if (lane_id == 0) {
        smem[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = lane_id < num_warp ? smem[lane_id] : 0;
        sum = _warp_shuffle_reduce<num_warp>(sum);
        if (tid == 0) {
            atomicAdd(b, sum);
        }
    }
}

template <const int block_size = 256>
__global__ void reduce_sum_i8x16_packed_kernel(int8_t *a, int32_t *b, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int32_t sum = 0;
    alignas(16) int8_t p[16];
    for (int i = idx * 16; i < N; i += gridDim.x * blockDim.x * 16) {
        LDST128BITS(p[0]) = LDST128BITS(a[i]);
#pragma unroll
        for (int t = 0; t < 16; t++) {
            sum += static_cast<int32_t>(p[t]);
        }
    }

    const int num_warp = (block_size + WARP_SIZE - 1) / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    __shared__ int32_t smem[num_warp];

    sum = _warp_shuffle_reduce<WARP_SIZE>(sum);
    if (lane_id == 0) {
        smem[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = lane_id < num_warp ? smem[lane_id] : 0;
        sum = _warp_shuffle_reduce<num_warp>(sum);
        if (tid == 0) {
            atomicAdd(b, sum);
        }
    }
}

template <const int block_size = 256>
__global__ void reduce_sum_i8x16_packed_dp4a_kernel(int8_t *a, int32_t *b, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int32_t sum = 0;
    const int beta = 0x01010101;
    int4 value;
    for (int i = idx * 16; i < N; i += gridDim.x * blockDim.x * 16) {
        LDST128BITS(value) = LDST128BITS(a[i]);
        sum = __dp4a(value.x, beta, sum);
        sum = __dp4a(value.y, beta, sum);
        sum = __dp4a(value.z, beta, sum);
        sum = __dp4a(value.w, beta, sum);
    }

    const int num_warp = (block_size + WARP_SIZE - 1) / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    __shared__ int32_t smem[num_warp];

    sum = _warp_shuffle_reduce<WARP_SIZE>(sum);
    if (lane_id == 0) {
        smem[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = lane_id < num_warp ? smem[lane_id] : 0;
        sum = _warp_shuffle_reduce<num_warp>(sum);
        if (tid == 0) {
            atomicAdd(b, sum);
        }
    }
}

__device__ __forceinline__ int _sum_i8x16(int4 value, int tsum) {
    const int beta = 0x01010101;
    tsum = __dp4a(value.x, beta, tsum);
    tsum = __dp4a(value.y, beta, tsum);
    tsum = __dp4a(value.z, beta, tsum);
    tsum = __dp4a(value.w, beta, tsum);
    return tsum;
}

template <const int block_size = 256>
__global__ void reduce_sum_i8x64_packed_dp4a_kernel(int8_t *a, int32_t *b, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int32_t tsum[4] = {0};
    int4 value[4];
    for (int i = idx * 64; i < N; i += gridDim.x * blockDim.x * 64) {
        LDST128BITS(value[0]) = LDST128BITS(a[i]);
        LDST128BITS(value[1]) = LDST128BITS(a[i + 16]);
        LDST128BITS(value[2]) = LDST128BITS(a[i + 32]);
        LDST128BITS(value[3]) = LDST128BITS(a[i + 48]);
#pragma unroll
        for (int j = 0; j < 4; j++)
            tsum[j] = _sum_i8x16(value[j], tsum[j]);
    }
    int sum = tsum[0] + tsum[1] + tsum[2] + tsum[3];

    const int num_warp = (block_size + WARP_SIZE - 1) / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    __shared__ int32_t smem[num_warp];

    sum = _warp_shuffle_reduce<WARP_SIZE>(sum);
    if (lane_id == 0) {
        smem[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = lane_id < num_warp ? smem[lane_id] : 0;
        sum = _warp_shuffle_reduce<num_warp>(sum);
        if (tid == 0) {
            atomicAdd(b, sum);
        }
    }
}

struct Generic {};

#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

#define binding_func_gen(name, num, element_dtype)                                                                     \
    torch::Tensor name(torch::Tensor a) {                                                                              \
        CHECK_T(a);                                                                                                    \
        const int total_elements = a.numel();                                                                          \
        const int threads_per_block = 256;                                                                             \
        const int blocks_per_grid = std::min((total_elements / num + threads_per_block - 1) / threads_per_block, 512); \
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
                    name##_kernel<threads_per_block>                                                                   \
                        <<<blocks_per_grid, threads_per_block, 0, stream>>>(reinterpret_cast<CastT *>(a.data_ptr()),   \
                                                                            reinterpret_cast<float *>(out.data_ptr()), \
                                                                            total_elements);                           \
                } else {                                                                                               \
                    TORCH_CHECK(false, #name " does not support " + string(toString(a.scalar_type())));                \
                }                                                                                                      \
            }));                                                                                                       \
        return out.to(a.scalar_type());                                                                                \
    }

#define binding_func_gen_int(name, num, element_dtype)                                                                 \
    torch::Tensor name(torch::Tensor a) {                                                                              \
        CHECK_T(a);                                                                                                    \
        const int total_elements = a.numel();                                                                          \
        const int threads_per_block = 256;                                                                             \
        const int blocks_per_grid =                                                                                    \
            std::min((total_elements / num + threads_per_block - 1) / threads_per_block, 1024);                        \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
        auto out = torch::zeros({1}, a.options().dtype(torch::kInt32));                                                \
        name##_kernel<threads_per_block><<<blocks_per_grid, threads_per_block, 0, stream>>>(                           \
            reinterpret_cast<int8_t *>(a.data_ptr()), reinterpret_cast<int *>(out.data_ptr()), total_elements);        \
        return out;                                                                                                    \
    }

binding_func_gen(reduce_sum, 1, Generic);
binding_func_gen(reduce_sum_fp32x4, 4, float);
binding_func_gen(reduce_sum_fp16x2, 2, half);
binding_func_gen(reduce_sum_fp16x8_packed, 8, half);
binding_func_gen_int(reduce_sum_i8, 1, int);
binding_func_gen_int(reduce_sum_i8x16_packed, 16, int);
binding_func_gen_int(reduce_sum_i8x16_packed_dp4a, 16, int);
binding_func_gen_int(reduce_sum_i8x64_packed_dp4a, 64, int);

// binding
#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(reduce_sum);
    torch_pybinding_func(reduce_sum_fp32x4);
    torch_pybinding_func(reduce_sum_fp16x2);
    torch_pybinding_func(reduce_sum_fp16x8_packed);
    torch_pybinding_func(reduce_sum_i8);
    torch_pybinding_func(reduce_sum_i8x16_packed);
    torch_pybinding_func(reduce_sum_i8x16_packed_dp4a);
    torch_pybinding_func(reduce_sum_i8x64_packed_dp4a);
}
