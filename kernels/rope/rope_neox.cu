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

__global__ void rope_fp32x4_kernel(float *a, float *b, int head_dim) {
    int pos = blockIdx.x * 4;
    int tid = threadIdx.x;
    pos += tid * 4 / (head_dim >> 1);
    int f_idx = tid * 4 % (head_dim >> 1);
    int idx = pos * head_dim + f_idx;
    float4 x = LDST128BITS(a[idx]);
    float4 y = LDST128BITS(a[idx + (head_dim >> 1)]);
    float inv_freq[4], c[4], s[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        inv_freq[i] = 1.f / __powf(theta, (2.0f * (f_idx + i)) / head_dim);
        __sincosf(pos * inv_freq[i], &s[i], &c[i]);
    }

    float4 x_new, y_new;
    x_new.x = x.x * c[0] - y.x * s[0];
    x_new.y = x.y * c[1] - y.y * s[1];
    x_new.z = x.z * c[2] - y.z * s[2];
    x_new.w = x.w * c[3] - y.w * s[3];
    y_new.x = y.x * c[0] + x.x * s[0];
    y_new.y = y.y * c[1] + x.y * s[1];
    y_new.z = y.z * c[2] + x.z * s[2];
    y_new.w = y.w * c[3] + x.w * s[3];

    LDST128BITS(b[idx]) = x_new;
    LDST128BITS(b[idx + head_dim / 2]) = y_new;
}

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
            a.data_ptr<float>(), b.data_ptr<float>(), head_dim);                                                       \
    }

binding_func_gen(rope, 1, float);
binding_func_gen(rope_fp32x4, 4, float);
// binding
#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(rope);
    torch_pybinding_func(rope_fp32x4);
}
