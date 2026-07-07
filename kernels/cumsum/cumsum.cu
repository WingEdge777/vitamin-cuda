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
        if (lane_id >= offset)
            val += __shfl_up_sync(0xffffffff, val, offset);
    }
    return val;
}

//
template <const int BLOCK_SIZE = 256>
__global__ void cumsum_fp32_kernel(float *a, float *ws, float *b, int hidden_size) {
    int tid = threadIdx.x;
    int bx = blockIdx.x;
    int offset = (bx * BLOCK_SIZE + tid);
    int val = a[offset];
    // warp
    const int num_warp = BLOCK_SIZE / 32;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    val = _warp_cum_sum<32>(lane_id, 32);
    // block
    __shared__ smem[num_warp];
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
    if (warp_id > 0)
        val += smem[warp_id - 1];

    if (lane_id == BLOCK_SIZE - 1) {
        ws[bx] = val;
    }

    b[offset] = val +
}

template <const int BLOCK_SIZE = 256>
__global__ void cumsum_fp32x4_kernel(float *a, float *ws, float *b, int hidden_size) {}

#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

#define binding_func_gen(name, element_type)                                                                           \
    torch::Tensor name(torch::Tensor a, torch::Tensor b) {                                                             \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        const int N = a.size(0);                                                                                       \
        const int threads_per_block = 256;                                                                             \
        const int chunk_size = 256 * 4;                                                                                \
        const int grid = (N + chunk_size - 1) / chunk_size;                                                            \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
        auto ws = torch::empty_like(a);                                                                                \
                                                                                                                       \
        name##_kernel<threads_per_block, chunk_size>                                                                   \
            <<<blocks_per_grid, threads_per_block, 0, stream>>>(reinterpret_cast<float *>(a.data_ptr()),               \
                                                                reinterpret_cast<float *>(ws.data_ptr()),              \
                                                                reinterpret_cast<float *>(b.data_ptr()),               \
                                                                N);                                                    \
    }

binding_func_gen(cmsum, 1, float);
binding_func_gen(cmsum_fp32x4, 4, float);
// binding
#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(cmsum);
    torch_pybinding_func(cmsum_fp32x4);
}
