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

const float theta = 10000.f;

// rope neox
__global__ void rope_kernel(float *a, float *b, int head_dim) {
    int pos = blockIdx.x;
    int idx = pos * head_dim + threadIdx.x;
    float x = a[idx];
    float y = a[idx + (head_dim >> 1)];

    int f_idx = threadIdx.x % (head_dim >> 1);

    float inv_freq = 1.f / __powf(theta, (2.0f * f_idx) / head_dim);
    float angle = pos * inv_freq;

    float c, s;
    __sincosf(angle, &s, &c);

    float x_new = x * c - y * s;
    float y_new = y * c + x * s;
    b[idx] = x_new;
    b[idx + head_dim / 2] = y_new;
}

__global__ void rope_fp32x4_kernel(float *a, float *b, int head_dim) {}

#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

#define binding_func_gen(name, num, element_dtype)                                                                     \
    void name(torch::Tensor a, torch::Tensor b) {                                                                      \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        const int seq_len = a.size(0);                                                                                 \
        const int head_dim = a.size(1);                                                                                \
        const int threads_per_block = head_dim / 2;                                                                    \
        const int blocks_per_grid = seq_len / num;                                                                     \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
                                                                                                                       \
        name##_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(                                              \
            reinterpret_cast<float *>(a.data_ptr()), reinterpret_cast<float *>(b.data_ptr()), head_dim);               \
    }

binding_func_gen(rope, 1, float);
binding_func_gen(rope_fp32x4, 4, float);
// binding
#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(rope);
    torch_pybinding_func(rope_fp32x4);
}
