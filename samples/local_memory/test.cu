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

__device__ __noinline__ void scale_by_ptr(float4 *ptr) {
    half2 *h2_ptr = reinterpret_cast<half2 *>(ptr);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        h2_ptr[i] = __hadd2(h2_ptr[i], h2_ptr[i]);
    }
}

__device__ __noinline__ float4 scale_by_val(float4 val) {
    union {
        float4 f;
        half2 h[4];
    } tmp;
    tmp.f = val;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        tmp.h[i] = __hadd2(tmp.h[i], tmp.h[i]);
    }
    return tmp.f;
}

// 溢出，但实际没溢出
__global__ void load_fp16x8_native_kernel(half *input, half *output, int N) {
    const int idx = threadIdx.x * 8;
    if (idx >= N)
        return;
    half2 pack[4];
    FLOAT4(pack[0]) = FLOAT4(input[idx]); // ❌ 对pack取地址 → 强制溢出到 local memory

    FLOAT4(output[idx]) = FLOAT4(pack[0]); // ❌ 对pack取地址 → 强制溢出到 local memory
}

// bad示例：外部函数调用，指针逃逸，避免编译器优化回物理寄存器
__global__ void load_fp16x8_bad_kernel(half *input, half *output, int N) {
    const int idx = threadIdx.x * 8;
    if (idx >= N)
        return;
    half2 pack[4];
    FLOAT4(pack[0]) = FLOAT4(input[idx]);               // ❌ 对pack取地址 → 强制溢出到 local memory
    scale_by_ptr(reinterpret_cast<float4 *>(&pack[0])); // ❌ 对pack取地址 → 强制溢出到 local memory

    FLOAT4(output[idx]) = FLOAT4(pack[0]); // ❌ 对pack取地址 → 强制溢出到 local memory
}
// good示例
__global__ void load_fp16x8_good_kernel(half *input, half *output, int N) {
    const int idx = threadIdx.x * 8;
    if (idx >= N)
        return;
    float4 pack = FLOAT4(input[idx]); // ✅ 纯值拷贝，毫无指针痕迹
    pack = scale_by_val(pack);

    FLOAT4(output[idx]) = pack; // ✅ 纯值拷贝，毫无指针痕迹
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
        name##_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(                                              \
            reinterpret_cast<half *>(a.data_ptr()), reinterpret_cast<half *>(b.data_ptr()), N);                        \
    }

binding_func_gen(load_fp16x8_native, 4, half);
binding_func_gen(load_fp16x8_bad, 4, half);
binding_func_gen(load_fp16x8_good, 4, half);

// binding
#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(load_fp16x8_native);
    torch_pybinding_func(load_fp16x8_bad);
    torch_pybinding_func(load_fp16x8_good);
}
