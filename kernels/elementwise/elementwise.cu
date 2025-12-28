#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <torch/extension.h>
#include <torch/types.h>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// fp32
__global__ void elementwise_add_operator_kernel(float *a, float *b, float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void elementwise_add_operator_fp32x4_kernel(float *a, float *b, float *c, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    float4 a_val = FLOAT4(a[idx]);
    float4 b_val = FLOAT4(b[idx]);
    float4 c_val = make_float4(a_val.x + b_val.x, a_val.y + b_val.y, a_val.z + b_val.z, a_val.w + b_val.w);
    if (idx + 3 < N) {
        FLOAT4(c[idx]) = c_val;
    } else {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            if (idx + i < N)
                c[idx + i] = a[idx + i] + b[idx + i];
        }
    }
}

// fp16
__global__ void elementwise_add_operator_kernel(half *a, half *b, half *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

// fp16x2
__global__ void elementwise_add_operator_fp16x2_kernel(half *a, half *b, half *c, int N) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < N) {
        half2 a_val = HALF2(a[idx]);
        half2 b_val = HALF2(b[idx]);
        HALF2(c[idx]) = __hadd2(a_val, b_val);
    }
}

// fp16x8
__global__ void elementwise_add_operator_fp16x8_kernel(half *a, half *b, half *c, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half2 a_val_0 = HALF2(a[idx]);
    half2 a_val_1 = HALF2(a[idx + 2]);
    half2 a_val_2 = HALF2(a[idx + 4]);
    half2 a_val_3 = HALF2(a[idx + 6]);

    half2 b_val_0 = HALF2(b[idx]);
    half2 b_val_1 = HALF2(b[idx + 2]);
    half2 b_val_2 = HALF2(b[idx + 4]);
    half2 b_val_3 = HALF2(b[idx + 6]);
    if (idx < N) {
        HALF2(c[idx]) = __hadd2(a_val_0, b_val_0);
    }
    if (idx + 2 < N) {
        HALF2(c[idx + 2]) = __hadd2(a_val_1, b_val_1);
    }
    if (idx + 4 < N) {
        HALF2(c[idx + 4]) = __hadd2(a_val_2, b_val_2);
    }
    if (idx + 6 < N) {
        HALF2(c[idx + 6]) = __hadd2(a_val_3, b_val_3);
    }
}

__global__ void elementwise_add_operator_fp16x8_packed_kernel(half *a, half *b, half *c, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    alignas(16) half pack_a[8];
    alignas(16) half pack_b[8];
    alignas(16) half pack_c[8];
    LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
    LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]);
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
        HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));
    }
    if (idx + 7 < N) {
        LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]);
    } else {
#pragma unroll
        for (int i = 0; i < 8; i++) {
            if (idx + i < N)
                c[idx + i] = pack_c[i];
        }
    }
}

template <typename scalar_t>
void elementwise_add_launcher(const torch::Tensor a, const torch::Tensor b, torch::Tensor c) {
    const int total_elements = a.numel();
    const int threads_per_block = 256;
    const int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;
    elementwise_add_operator_kernel<<<blocks_per_grid, threads_per_block>>>(
        reinterpret_cast<scalar_t *>(a.data_ptr()), reinterpret_cast<scalar_t *>(b.data_ptr()),
        reinterpret_cast<scalar_t *>(c.data_ptr()), total_elements);
}

void elementwise_add(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    AT_DISPATCH_SWITCH(a.scalar_type(), "elementwise_add", AT_DISPATCH_CASE(at::kFloat, [&] {
                           elementwise_add_launcher<float>(a, b, c);
                       }) AT_DISPATCH_CASE(at::kHalf, [&] { elementwise_add_launcher<half>(a, b, c); }));
}

void elementwise_add_fp32x4(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const int total_elements = a.numel();
    const int threads_per_block = 256; // keep block thread num equal
    const int blocks_per_grid = (total_elements / 4 + threads_per_block - 1) / threads_per_block;
    elementwise_add_operator_fp32x4_kernel<<<blocks_per_grid, threads_per_block>>>(
        reinterpret_cast<float *>(a.data_ptr()), reinterpret_cast<float *>(b.data_ptr()),
        reinterpret_cast<float *>(c.data_ptr()), total_elements);
}

void elementwise_add_fp16x2(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const int total_elements = a.numel();
    const int threads_per_block = 256; // keep block thread num equal
    const int blocks_per_grid = (total_elements / 2 + threads_per_block - 1) / threads_per_block;
    elementwise_add_operator_fp16x2_kernel<<<blocks_per_grid, threads_per_block>>>(
        reinterpret_cast<half *>(a.data_ptr()), reinterpret_cast<half *>(b.data_ptr()),
        reinterpret_cast<half *>(c.data_ptr()), total_elements);
}

void elementwise_add_fp16x8(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const int total_elements = a.numel();
    const int threads_per_block = 256;
    const int blocks_per_grid = (total_elements / 8 + threads_per_block - 1) / threads_per_block;
    elementwise_add_operator_fp16x8_kernel<<<blocks_per_grid, threads_per_block>>>(
        reinterpret_cast<half *>(a.data_ptr()), reinterpret_cast<half *>(b.data_ptr()),
        reinterpret_cast<half *>(c.data_ptr()), total_elements);
}

void elementwise_add_fp16x8_packed(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const int total_elements = a.numel();
    const int threads_per_block = 256;
    const int blocks_per_grid = (total_elements / 8 + threads_per_block - 1) / threads_per_block;
    elementwise_add_operator_fp16x8_packed_kernel<<<blocks_per_grid, threads_per_block>>>(
        reinterpret_cast<half *>(a.data_ptr()), reinterpret_cast<half *>(b.data_ptr()),
        reinterpret_cast<half *>(c.data_ptr()), total_elements);
}

#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(elementwise_add);
    torch_pybinding_func(elementwise_add_fp32x4);
    torch_pybinding_func(elementwise_add_fp16x2);
    torch_pybinding_func(elementwise_add_fp16x8);
    torch_pybinding_func(elementwise_add_fp16x8_packed);
}