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

// fp32
template <const int warp_size = WARP_SIZE, typename T>
__device__ __forceinline__ T _warp_mask_reduce(T x) {
    T sum = 0.f;
#pragma unroll
    for (int mask = warp_size >> 1; mask > 0; mask >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, x, mask);
    }
    return sum;
}

template <const int block_size = 256, typename T>
__global__ void reduce_sum_kernel(T *a, T *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_warp = (block_size + warp_size - 1) / warp_size;
    if (idx < N) {
        b[idx] = _reduce_sum(a[idx]);
    }
}

struct Generic {};

#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

#define binding_func_gen(name, num, element_dtype)                                                                     \
    void name(torch::Tensor a) {                                                                                       \
        CHECK_T(a);                                                                                                    \
        const int total_elements = a.numel();                                                                          \
        const int threads_per_block = 256;                                                                             \
        const int blocks_per_grid = (total_elements / num + threads_per_block - 1) / threads_per_block;                \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
        auto options = torch::TensorOptions().dtype(y_th_type).device(torch::kCUDA, 0);                                \
        auto out = torch::zeros({1}, options);                                                                         \
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
                        reinterpret_cast<CastT *>(out.data_ptr()),                                                     \
                        total_elements);                                                                               \
                } else {                                                                                               \
                    TORCH_CHECK(false, #name " does not support " + string(toString(b.scalar_type())));                \
                }                                                                                                      \
            }));                                                                                                       \
        cudaStreamSynchronize(stream);                                                                                 \
        return out;
}

binding_func_gen(reduce_sum, 1, Generic);
// binding_func_gen(reduce_sum_fp32x4, 4, float);
// binding_func_gen(reduce_sum_fp16x2, 2, half);
// binding_func_gen(reduce_sum_fp16x8, 8, half);
// binding_func_gen(reduce_sum_fp16x8_packed, 8, half);
// binding
#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(reduce_sum);
    // torch_pybinding_func(reduce_sum_fp32x4);
    // torch_pybinding_func(reduce_sum_fp16x2);
    // torch_pybinding_func(reduce_sum_fp16x8);
    // torch_pybinding_func(reduce_sum_fp16x8_packed);
}
