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
__device__ __forceinline__ float _warp_cum_sum(int lane_id, float val) {
#pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float other_val = __shfl_up_sync(0xffffffff, val, offset);
        if (lane_id >= offset) {
            val += other_val;
        }
    }
    return val;
}

// 1 cta for one row
template <const int BLOCK_SIZE = 256, const int CHUNK_SIZE = 206>
__global__ void cumsum_fp32_kernel(float *a, float *b, int n) {
    int tid = threadIdx.x;
    int row = blockIdx.x;

    const int num_warp = BLOCK_SIZE / 32;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    __shared__ float smem[num_warp];
    __shared__ float block_delta;

    float delta = 0.f;
    int row_offset = row * n;
    for (int i = 0; i < n; i += BLOCK_SIZE) {
        int col = i + tid;
        float val = (col < n) ? a[row_offset + col] : 0.0f;
        // warp
        val = _warp_cum_sum<32>(lane_id, val);

        // block
        if (lane_id == 32 - 1)
            smem[warp_id] = val;
        __syncthreads();

        if (warp_id == 0) {
            float tmp_v = tid < num_warp ? smem[tid] : 0.f;
            tmp_v = _warp_cum_sum(lane_id, tmp_v);
            if (tid < num_warp)
                smem[tid] = tmp_v;
        }
        __syncthreads();

        if (warp_id > 0) {
            val += smem[warp_id - 1];
        }

        val += delta;

        if (col < n) {
            b[row_offset + col] = val;
        }

        if (tid == BLOCK_SIZE - 1) {
            block_delta = val;
        }
        __syncthreads();

        delta = block_delta;
    }
}

template <const int BLOCK_SIZE = 256, const int CHUNK_SIZE = 2048>
__global__ void cumsum_fp32x4_kernel(float *a, float *b, int n) {}

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

binding_func_gen(cumsum_fp32, 1, float);
binding_func_gen(cumsum_fp32x4, 4, float);
// binding
#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(cumsum_fp32);
    torch_pybinding_func(cumsum_fp32x4);
}
