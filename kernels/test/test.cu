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

const int WARP_SIZE = 32;

// load fp32
__global__ void load_fp32x4_kernel(float *a, float *b, int n) {
    __shared__ float s[128];

    FLOAT4(s[threadIdx.x * 4]) = FLOAT4(a[threadIdx.x * 4]);
    FLOAT4(b[threadIdx.x * 4]) = FLOAT4(s[threadIdx.x * 4]);
    __syncthreads();
    s[threadIdx.x] *= a[threadIdx.x] * 2;
    __syncthreads();
    b[threadIdx.x] = s[threadIdx.x];
}

#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

#define binding_func_gen(name, num, element_dtype)                                                                     \
    void name(torch::Tensor a, torch::Tensor b) {                                                                      \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        const int N = a.size(0);                                                                                       \
        const dim3 threads_per_block(32);                                                                              \
        const dim3 blocks_per_grid(N / 32 / num);                                                                      \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
                                                                                                                       \
        name##_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(a.data_ptr<float>(), b.data_ptr<float>(), N); \
    }

binding_func_gen(load_fp32x4, 4, float);

// binding
#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { torch_pybinding_func(load_fp32x4); }
