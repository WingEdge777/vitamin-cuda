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

struct __align__(8) MD {
    float m;
    float d;
};

template <const int warp_size = WARP_SIZE>
__device__ __forceinline__ MD _warp_online_softmax_reduce(MD val) {
#pragma unroll
    for (int offset = warp_size >> 1; offset > 0; offset >>= 1) {
        float other_m = __shfl_xor_sync(0xffffffff, val.m, offset);
        float other_d = __shfl_xor_sync(0xffffffff, val.d, offset);

        float new_m = fmaxf(val.m, other_m);
        val.d = val.d * __expf(val.m - new_m) + other_d * __expf(other_m - new_m);
        val.m = new_m;
    }
    return val;
}

// safe softmax: a's shape [bs, hidden_size], support hiddensize <= 8192
template <const int BLOCK_SIZE = 256>
__global__ void softmax_kernel(float *a, float *b, int hidden_size) {
    int row_offset = blockIdx.x * hidden_size;
    int tid = threadIdx.x;

    float in[32];
    MD val{-INFINITY, 0.f};
    int i = 0;
    for (; tid + i * BLOCK_SIZE < hidden_size; i++) {
        in[i] = a[row_offset + tid + i * BLOCK_SIZE];
        val.m = fmaxf(val.m, in[i]);
    }
    int cnt = i;
#pragma unroll
    for (int i = 0; i < cnt; i++) {
        val.d += __expf(in[i] - val.m);
    }
    val = _warp_online_softmax_reduce<WARP_SIZE>(val);
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    const int num_warps = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ MD sdata[num_warps];

    if (lane_id == 0) {
        sdata[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = lane_id < num_warps ? sdata[lane_id] : MD{-INFINITY, 0.f};
        val = _warp_online_softmax_reduce<num_warps>(val);
        if (lane_id == 0) {
            sdata[0] = val;
        }
    }
    __syncthreads();
    val = sdata[0];

#pragma unroll
    for (int i = 0; i < cnt; i++) {
        int offset = tid + i * BLOCK_SIZE;
        b[row_offset + offset] = __expf(in[i] - val.m) / val.d;
    }
}

template <const int BLOCK_SIZE = 256>
__global__ void softmax_fp32x4_kernel(float *a, float *b, int hidden_size) {
    int row_offset = blockIdx.x * hidden_size;
    int tid = threadIdx.x;

    float4 in[8];
    MD val{-INFINITY, 0.f};

    int i = 0;
#pragma unroll
    for (; (tid + i * BLOCK_SIZE) * 4 < hidden_size; i++) {
        in[i] = LDST128BITS(a[row_offset + (tid + i * BLOCK_SIZE) * 4]);
        val.m = fmaxf(fmaxf(in[i].x, in[i].y), fmaxf(in[i].z, in[i].w));
    }
    int cnt = i;
#pragma unroll
    for (int i = 0; i < cnt; i++) {
        val.d += __expf(in[i].x - val.m) + __expf(in[i].y - val.m) + __expf(in[i].z - val.m) + __expf(in[i].w - val.m);
    }
    val = _warp_online_softmax_reduce<WARP_SIZE>(val);
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    const int num_warps = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ MD sdata[num_warps];

    if (lane_id == 0) {
        sdata[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = lane_id < num_warps ? sdata[lane_id] : MD{-INFINITY, 0.f};
        val = _warp_online_softmax_reduce<num_warps>(val);
        if (lane_id == 0) {
            sdata[0] = val;
        }
    }
    __syncthreads();
    val = sdata[0];

#pragma unroll
    for (int i = 0; i < cnt; i++) {
        float4 tmp;
        tmp.x = __expf(in[i].x - val.m) / val.d;
        tmp.y = __expf(in[i].y - val.m) / val.d;
        tmp.z = __expf(in[i].z - val.m) / val.d;
        tmp.w = __expf(in[i].w - val.m) / val.d;

        int offset = tid + i * BLOCK_SIZE;
        LDST128BITS(b[row_offset + offset * 4]) = tmp;
    }
}

#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

#define binding_func_gen(name, num)                                                                                    \
    void name(torch::Tensor a, torch::Tensor b) {                                                                      \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        const int bs = a.size(0);                                                                                      \
        const int lda = a.size(1);                                                                                     \
        const int threads_per_block = 256;                                                                             \
        const int blocks_per_grid = bs;                                                                                \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
                                                                                                                       \
        switch (a.scalar_type()) {                                                                                     \
            case torch::kFloat:                                                                                        \
                name##_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(                                      \
                    reinterpret_cast<float *>(a.data_ptr()), reinterpret_cast<float *>(b.data_ptr()), lda);            \
                break;                                                                                                 \
            default: AT_ERROR(#name " not implemented for '", toString(a.scalar_type()), "'");                         \
        }                                                                                                              \
    }

binding_func_gen(softmax, 1);
binding_func_gen(softmax_fp32x4, 4);

// binding
#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(softmax);
    torch_pybinding_func(softmax_fp32x4);
}
