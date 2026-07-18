#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/types.h>

#include "../common/pack.cuh"

template <const int warp_size = WARP_SIZE>
__device__ __forceinline__ void _warp_reduce(float &val, float &square) {
#pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
        square += __shfl_xor_sync(0xffffffff, square, offset);
    }
}

// 1 cta for one row
template <const int BLOCK_SIZE = 256, const int CHUNK_SIZE = 256>
__global__ void norm_fp32_kernel(float *a, float *b, int n) {
    int tid = threadIdx.x;
    int row = blockIdx.x;

    const int num_warp = BLOCK_SIZE / 32;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    __shared__ float s1[num_warp], s2[num_warp];
    __shared__ float mean_s, std_s;

    int row_offset = row * n;
    float val = 0.f, square = 0.f;
    for (int i = 0; i < n; i += BLOCK_SIZE) {
        int col = i + tid;
        float tmp = (col < n) ? a[row_offset + col] : 0.0f;
        val += tmp;
        square += tmp * tmp;
    }
    _warp_reduce(val, square);
    if (lane_id == 0) {
        s1[warp_id] = val;
        s2[warp_id] = square;
    }
    __syncthreads();
    if (warp_id == 0) {
        float x = tid < num_warp ? s1[tid] : 0.f;
        float sq = tid < num_warp ? s2[tid] : 0.f;
        _warp_reduce<num_warp>(x, sq);
        if (tid == 0) {
            mean_s = x / n;
            std_s = sqrtf(sq / n - mean_s * mean_s);
        }
    }
    __syncthreads();
    float mean = mean_s;
    float std = std_s;

    for (int i = 0; i < n; i += BLOCK_SIZE) {
        int col = i + tid;
        if (col < n) {
            b[row_offset + col] = (a[row_offset + col] - mean) / std;
        }
    }
}

template <const int BLOCK_SIZE = 256, const int CHUNK_SIZE = 1024>
__global__ void norm_fp32x4_kernel(float *a, float *b, int n) {
    int tid = threadIdx.x;
    int row = blockIdx.x;

    const int num_warp = BLOCK_SIZE / 32;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    __shared__ float s1[num_warp], s2[num_warp];
    __shared__ float mean_s, std_s;

    int row_offset = row * n;
    float val = 0.f, square = 0.f;
    for (int i = 0; i < n; i += CHUNK_SIZE) {
        int col = i + tid * 4;
        float4 tmp = (col < n) ? FLOAT4(a[row_offset + col]) : make_float4(0.f, 0.f, 0.f, 0.f);
        val += tmp.x + tmp.y + tmp.z + tmp.w;
        square += tmp.x * tmp.x + tmp.y * tmp.y + tmp.z * tmp.z + tmp.w * tmp.w;
    }
    _warp_reduce(val, square);
    if (lane_id == 0) {
        s1[warp_id] = val;
        s2[warp_id] = square;
    }
    __syncthreads();
    if (warp_id == 0) {
        float x = tid < num_warp ? s1[tid] : 0.f;
        float sq = tid < num_warp ? s2[tid] : 0.f;
        _warp_reduce<num_warp>(x, sq);
        if (tid == 0) {
            mean_s = x / n;
            std_s = sqrtf(sq / n - mean_s * mean_s);
        }
    }
    __syncthreads();
    float mean = mean_s;
    float std = std_s;

    for (int i = 0; i < n; i += CHUNK_SIZE) {
        int col = i + tid * 4;
        if (col < n) {
            float4 tmp = FLOAT4(a[row_offset + col]);
            float4 out;
            out.x = (tmp.x - mean) / std;
            out.y = (tmp.y - mean) / std;
            out.z = (tmp.z - mean) / std;
            out.w = (tmp.w - mean) / std;
            FLOAT4(b[row_offset + col]) = out;
        }
    }
}

#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

#define binding_func_gen(name, num, element_type)                                                                      \
    void name(torch::Tensor a, torch::Tensor b) {                                                                      \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        const int bs = a.size(0);                                                                                      \
        const int lda = a.size(1);                                                                                     \
        const int threads_per_block = 256;                                                                             \
        const int chunk_size = threads_per_block * num;                                                                \
        const int grid = bs;                                                                                           \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
                                                                                                                       \
        name##_kernel<threads_per_block, chunk_size><<<grid, threads_per_block, 0, stream>>>(                          \
            reinterpret_cast<float *>(a.data_ptr()), reinterpret_cast<float *>(b.data_ptr()), lda);                    \
    }

binding_func_gen(norm_fp32, 1, float);
binding_func_gen(norm_fp32x4, 4, float);
// binding
#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(norm_fp32);
    torch_pybinding_func(norm_fp32x4);
}
