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

__device__ float _sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ half _sigmoid(half x) { return __hdiv(__float2half(1.0f), __hadd(__float2half(1.0f), hexp(__hneg(x)))); }

// fp32

__global__ void sigmoid_operator_kernel(float *a, float *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        b[idx] = _sigmoid(a[idx]);
    }
}

__global__ void sigmoid_operator_fp32x4_kernel(float *a, float *b, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    float4 a_val = FLOAT4(a[idx]);
    if (idx + 3 < N) {
        float4 b_val = make_float4(_sigmoid(a_val.x), _sigmoid(a_val.y), _sigmoid(a_val.z), _sigmoid(a_val.w));
        FLOAT4(b[idx]) = b_val;
    } else {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            if (idx + i < N)
                b[idx + i] = _sigmoid(a[idx + i]);
        }
    }
}

// fp16
__global__ void sigmoid_operator_kernel(half *a, half *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        b[idx] = _sigmoid(a[idx]);
    }
}

// fp16x2
__global__ void sigmoid_operator_fp16x2_kernel(half *a, half *b, int N) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx + 1 < N) {
        half2 a_val = HALF2(a[idx]);
        half2 b_val = make_half2(_sigmoid(a_val.x), _sigmoid(a_val.y));
        HALF2(b[idx]) = b_val;
    } else {
        b[idx] = _sigmoid(a[idx]);
    }
}

// fp16x8
__global__ void sigmoid_operator_fp16x8_kernel(half *a, half *b, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half2 a_val_0 = HALF2(a[idx]);
    half2 a_val_1 = HALF2(a[idx + 2]);
    half2 a_val_2 = HALF2(a[idx + 4]);
    half2 a_val_3 = HALF2(a[idx + 6]);

    if (idx < N) {
        HALF2(b[idx]) = make_half2(_sigmoid(a_val_0.x), _sigmoid(a_val_0.y));
    }
    if (idx + 2 < N) {
        HALF2(b[idx + 2]) = make_half2(_sigmoid(a_val_1.x), _sigmoid(a_val_1.y));
    }
    if (idx + 4 < N) {
        HALF2(b[idx + 4]) = make_half2(_sigmoid(a_val_2.x), _sigmoid(a_val_2.y));
    }
    if (idx + 6 < N) {
        HALF2(b[idx + 6]) = make_half2(_sigmoid(a_val_3.x), _sigmoid(a_val_3.y));
    }
}

// fp16x8 packed r/w
__global__ void sigmoid_operator_fp16x8_packed_kernel(half *a, half *b, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    alignas(16) half pack_a[8];
    alignas(16) half pack_b[8];
    LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        pack_b[i] = _sigmoid(pack_a[i]);
    }
    if (idx + 7 < N) {
        LDST128BITS(b[idx]) = LDST128BITS(pack_b[0]);
    } else {
#pragma unroll
        for (int i = 0; i < 8; i++) {
            if (idx + i < N)
                b[idx + i] = pack_b[i];
        }
    }
}

template <typename scalar_t>
void sigmoid_launcher(const torch::Tensor a, const torch::Tensor b) {
    const int total_elements = a.numel();
    const int threads_per_block = 256;
    const int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;
    sigmoid_operator_kernel<<<blocks_per_grid, threads_per_block>>>(
        reinterpret_cast<scalar_t *>(a.data_ptr()), reinterpret_cast<scalar_t *>(b.data_ptr()), total_elements);
}

// func for binding
void sigmoid(torch::Tensor a, torch::Tensor b) {
    AT_DISPATCH_SWITCH(a.scalar_type(), "sigmoid", AT_DISPATCH_CASE(at::kFloat, [&] {
                           sigmoid_launcher<float>(a, b);
                       }) AT_DISPATCH_CASE(at::kHalf, [&] { sigmoid_launcher<half>(a, b); }));
}

void sigmoid_fp32x4(torch::Tensor a, torch::Tensor b) {
    const int total_elements = a.numel();
    const int threads_per_block = 256; // keep block thread num equal
    const int blocks_per_grid = (total_elements / 4 + threads_per_block - 1) / threads_per_block;
    sigmoid_operator_fp32x4_kernel<<<blocks_per_grid, threads_per_block>>>(
        reinterpret_cast<float *>(a.data_ptr()), reinterpret_cast<float *>(b.data_ptr()), total_elements);
}

void sigmoid_fp16x2(torch::Tensor a, torch::Tensor b) {
    const int total_elements = a.numel();
    const int threads_per_block = 256; // keep block thread num equal
    const int blocks_per_grid = (total_elements / 2 + threads_per_block - 1) / threads_per_block;
    sigmoid_operator_fp16x2_kernel<<<blocks_per_grid, threads_per_block>>>(
        reinterpret_cast<half *>(a.data_ptr()), reinterpret_cast<half *>(b.data_ptr()), total_elements);
}

void sigmoid_fp16x8(torch::Tensor a, torch::Tensor b) {
    const int total_elements = a.numel();
    const int threads_per_block = 256;
    const int blocks_per_grid = (total_elements / 8 + threads_per_block - 1) / threads_per_block;
    sigmoid_operator_fp16x8_kernel<<<blocks_per_grid, threads_per_block>>>(
        reinterpret_cast<half *>(a.data_ptr()), reinterpret_cast<half *>(b.data_ptr()), total_elements);
}

void sigmoid_fp16x8_packed(torch::Tensor a, torch::Tensor b) {
    const int total_elements = a.numel();
    const int threads_per_block = 256;
    const int blocks_per_grid = (total_elements / 8 + threads_per_block - 1) / threads_per_block;
    sigmoid_operator_fp16x8_packed_kernel<<<blocks_per_grid, threads_per_block>>>(
        reinterpret_cast<half *>(a.data_ptr()), reinterpret_cast<half *>(b.data_ptr()), total_elements);
}

// binding
#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(sigmoid);
    torch_pybinding_func(sigmoid_fp32x4);
    torch_pybinding_func(sigmoid_fp16x2);
    torch_pybinding_func(sigmoid_fp16x8);
    torch_pybinding_func(sigmoid_fp16x8_packed);
}