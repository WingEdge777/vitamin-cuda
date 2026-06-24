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

template <const int BLOCK_SIZE = 256, typename T>
__global__ void sort_kernel(T *x, T *y, int n) {}

#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

#define binding_func_gen(name, block_size)                                                                             \
    void name(torch::Tensor a, torch::Tensor b) {                                                                      \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        const int N = a.size(0);                                                                                       \
        const dim3 blocks_per_grid(N / block_size);                                                                    \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
                                                                                                                       \
        if (a.dtype() == torch::kHalf) {                                                                               \
            name##_kernel<block_size><<<blocks_per_grid, block_size, 0, stream>>>(                                     \
                reinterpret_cast<__half *>(a.data_ptr()), reinterpret_cast<__half *>(b.data_ptr()), N);                \
        } else {                                                                                                       \
            name##_kernel<block_size><<<blocks_per_grid, block_size, 0, stream>>>(                                     \
                reinterpret_cast<__nv_bfloat16 *>(a.data_ptr()), reinterpret_cast<__nv_bfloat16 *>(b.data_ptr()), N);  \
        }                                                                                                              \
    }

binding_func_gen(sort, 256);
extern void cub_sort(torch::Tensor a, torch::Tensor b);

#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(cub_sort);
    torch_pybinding_func(sort);
}
