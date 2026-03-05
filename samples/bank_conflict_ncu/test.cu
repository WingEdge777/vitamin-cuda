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
    __shared__ float s[512];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float4 val = FLOAT4(a[idx * 4]);
    val.x *= 2;
    FLOAT4(s[threadIdx.x * 4]) = val;
    __syncthreads();

    FLOAT4(b[idx * 4]) = FLOAT4(s[threadIdx.x * 4]);
}

#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

#define binding_func_gen(name, num, element_dtype)                                                                     \
    void name(torch::Tensor a, torch::Tensor b) {                                                                      \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        const int N = a.size(0);                                                                                       \
        const int threads_per_block = 128;                                                                             \
        const dim3 blocks_per_grid = N / num / threads_per_block;                                                      \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
                                                                                                                       \
        name##_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(a.data_ptr<float>(), b.data_ptr<float>(), N); \
    }

binding_func_gen(load_fp32x4, 4, float);

// binding
#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { torch_pybinding_func(load_fp32x4); }
