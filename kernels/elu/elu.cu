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

const float alpha = 1.0f;

__device__ __forceinline__ float _elu(float x) { return x > 0.f ? x : alpha * (expf(x) - 1.f); }

__device__ __forceinline__ half _elu(half x) {
    const half zero = __float2half(0.f);
    return __hgt(x, zero) ? x : __hmul(__float2half(alpha), __hsub(hexp(x), __float2half(1.f)));
}

__device__ __forceinline__ half2 _elu(half2 x) {
    const half2 zero2 = __float2half2_rn(0.0f);
    const half2 one2 = __float2half2_rn(1.0f);
    const half zero = __float2half(0.0f);
    const half2 alpha2 = __float2half2_rn(alpha);

    half2 exp_part = __hmul2(alpha2, __hsub2(h2exp(x), one2));

    half x_lo = __low2half(x);
    half x_hi = __high2half(x);

    half exp_lo = __low2half(exp_part);
    half exp_hi = __high2half(exp_part);

    return make_half2(__hgt(x_lo, zero) ? x_lo : exp_lo, __hgt(x_hi, zero) ? x_hi : exp_hi);
}

// fp32

__global__ void elu_operator_kernel(float *a, float *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        b[idx] = _elu(a[idx]);
    }
}

__global__ void elu_operator_fp32x4_kernel(float *a, float *b, int N) {
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    float4 a_val = FLOAT4(a[idx]);
    if (idx + 3 < N) {
        float4 b_val = make_float4(_elu(a_val.x), _elu(a_val.y), _elu(a_val.z), _elu(a_val.w));
        FLOAT4(b[idx]) = b_val;
    } else {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            if (idx + i < N)
                b[idx + i] = _elu(a[idx + i]);
        }
    }
}

// fp16
__global__ void elu_operator_kernel(half *a, half *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        b[idx] = _elu(a[idx]);
    }
}

// fp16x2
__global__ void elu_operator_fp16x2_kernel(half *a, half *b, int N) {
    int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    HALF2(b[idx]) = _elu(HALF2(a[idx]));
}

// fp16x8
__global__ void elu_operator_fp16x8_kernel(half *a, half *b, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half2 a_val_0 = HALF2(a[idx]);
    half2 a_val_1 = HALF2(a[idx + 2]);
    half2 a_val_2 = HALF2(a[idx + 4]);
    half2 a_val_3 = HALF2(a[idx + 6]);

    if (idx < N) {
        HALF2(b[idx]) = _elu(a_val_0);
    }
    if (idx + 2 < N) {
        HALF2(b[idx + 2]) = _elu(a_val_1);
    }
    if (idx + 4 < N) {
        HALF2(b[idx + 4]) = _elu(a_val_2);
    }
    if (idx + 6 < N) {
        HALF2(b[idx + 6]) = _elu(a_val_3);
    }
}

// fp16x8 packed r/w
__global__ void elu_operator_fp16x8_packed_kernel(half *a, half *b, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx + 7 < N) {
        alignas(16) half pack_a[8];
        alignas(16) half pack_b[8];

        LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);

#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            HALF2(pack_b[i]) = _elu(HALF2(pack_a[i]));
        }

        LDST128BITS(b[idx]) = LDST128BITS(pack_b[0]);
    } else {
#pragma unroll
        for (int i = 0; i < 8; i++) {
            if (idx + i < N)
                b[idx + i] = _elu(a[idx + i]);
        }
    }
}

template <typename scalar_t>
void elu_launcher(const torch::Tensor a, const torch::Tensor b) {
    const int total_elements = a.numel();
    const int threads_per_block = 256;
    const int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;
    elu_operator_kernel<<<blocks_per_grid, threads_per_block>>>(
        reinterpret_cast<scalar_t *>(a.data_ptr()), reinterpret_cast<scalar_t *>(b.data_ptr()), total_elements);
}

// func for binding
void elu(torch::Tensor a, torch::Tensor b) {
    AT_DISPATCH_SWITCH(a.scalar_type(), "elu", AT_DISPATCH_CASE(at::kFloat, [&] {
                           elu_launcher<float>(a, b);
                       }) AT_DISPATCH_CASE(at::kHalf, [&] { elu_launcher<half>(a, b); }));
}

void elu_fp32x4(torch::Tensor a, torch::Tensor b) {
    const int total_elements = a.numel();
    const int threads_per_block = 256; // keep block thread num equal
    const int blocks_per_grid = (total_elements / 4 + threads_per_block - 1) / threads_per_block;
    elu_operator_fp32x4_kernel<<<blocks_per_grid, threads_per_block>>>(
        reinterpret_cast<float *>(a.data_ptr()), reinterpret_cast<float *>(b.data_ptr()), total_elements);
}

void elu_fp16x2(torch::Tensor a, torch::Tensor b) {
    const int total_elements = a.numel();
    const int threads_per_block = 256; // keep block thread num equal
    const int blocks_per_grid = (total_elements / 2 + threads_per_block - 1) / threads_per_block;
    elu_operator_fp16x2_kernel<<<blocks_per_grid, threads_per_block>>>(
        reinterpret_cast<half *>(a.data_ptr()), reinterpret_cast<half *>(b.data_ptr()), total_elements);
}

void elu_fp16x8(torch::Tensor a, torch::Tensor b) {
    const int total_elements = a.numel();
    const int threads_per_block = 256;
    const int blocks_per_grid = (total_elements / 8 + threads_per_block - 1) / threads_per_block;
    elu_operator_fp16x8_kernel<<<blocks_per_grid, threads_per_block>>>(
        reinterpret_cast<half *>(a.data_ptr()), reinterpret_cast<half *>(b.data_ptr()), total_elements);
}

void elu_fp16x8_packed(torch::Tensor a, torch::Tensor b) {
    const int total_elements = a.numel();
    const int threads_per_block = 256;
    const int blocks_per_grid = (total_elements / 8 + threads_per_block - 1) / threads_per_block;
    elu_operator_fp16x8_packed_kernel<<<blocks_per_grid, threads_per_block>>>(
        reinterpret_cast<half *>(a.data_ptr()), reinterpret_cast<half *>(b.data_ptr()), total_elements);
}

// binding
#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(elu);
    torch_pybinding_func(elu_fp32x4);
    torch_pybinding_func(elu_fp16x2);
    torch_pybinding_func(elu_fp16x8);
    torch_pybinding_func(elu_fp16x8_packed);
}