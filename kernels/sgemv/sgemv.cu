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

template <const int warp_size = WARP_SIZE>
__device__ __forceinline__ float _warp_shuffle_reduce_sum(float val) {
#pragma unroll
    for (int offset = warp_size >> 1; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

template <const int num_warp = 8>
__device__ __forceinline__ float _block_reduce_sum(float val) {
    val = _warp_shuffle_reduce_sum<WARP_SIZE>(val);

    __shared__ float sdata[num_warp];
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    if (lane_id == 0) {
        sdata[warp_id] = val;
    }
    __syncthreads();
    if (warp_id == 0) {
        val = lane_id < num_warp ? sdata[lane_id] : 0.f;
        val = _warp_shuffle_reduce_sum<num_warp>(val);
        if (lane_id == 0) {
            sdata[0] = val;
        }
    }
    __syncthreads();
    return sdata[0];
}

// gemv fp32
template <const int BLOCK_SIZE = 128>
__global__ void gemv_kernel(float *a, float *b, float *c, int out_channels, int in_channels) {
    int x = threadIdx.x;
    int y = blockIdx.x;

    float sum = 0.0f;
    if (y < out_channels) {
        for (int i = x; i < in_channels; i += blockDim.x) {
            sum += a[y * in_channels + i] * b[i];
        }
        sum = _block_reduce_sum<BLOCK_SIZE / WARP_SIZE>(sum);
        if (x == 0) {
            c[y] = sum;
        }
    }
}

// gemv fp32x4
template <const int BLOCK_SIZE = 128>
__global__ void gemv_fp32x4_kernel(float *a, float *b, float *c, int out_channels, int in_channels) {
    const int ROWS_PER_BLOCK = 4;

    int tid = threadIdx.x;
    int row_start = blockIdx.x * ROWS_PER_BLOCK;

    float sum[ROWS_PER_BLOCK] = {0.0f};

    for (int k = tid * 4; k < in_channels; k += blockDim.x * 4) {
        float4 vb = LDST128BITS(b[k]);
#pragma unroll
        for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
            int current_row = row_start + r;
            float4 va = LDST128BITS(a[current_row * in_channels + k]);
            sum[r] += va.x * vb.x;
            sum[r] += va.y * vb.y;
            sum[r] += va.z * vb.z;
            sum[r] += va.w * vb.w;
        }
    }
    float c_out[ROWS_PER_BLOCK] = {0.0f};
#pragma unroll
    for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
        int current_row = row_start + r;
        float res = _block_reduce_sum<BLOCK_SIZE / WARP_SIZE>(sum[r]);
        if (tid == 0) {
            c_out[r] = res;
        }
    }
    if (tid == 0) {
        LDST128BITS(c[row_start]) = LDST128BITS(c_out[0]);
    }
}

#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

#define binding_func_gen(name, num, element_dtype)                                                                     \
    void name(torch::Tensor a, torch::Tensor b, torch::Tensor c) {                                                     \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        const int out_channels = a.size(0);                                                                            \
        const int in_channels = a.size(1);                                                                             \
        const int threads_per_block = 128;                                                                             \
        const int blocks_per_grid = (out_channels + num - 1) / num;                                                    \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
                                                                                                                       \
        name##_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(reinterpret_cast<float *>(a.data_ptr()),      \
                                                                         reinterpret_cast<float *>(b.data_ptr()),      \
                                                                         reinterpret_cast<float *>(c.data_ptr()),      \
                                                                         out_channels,                                 \
                                                                         in_channels);                                 \
    }

binding_func_gen(gemv, 1, float);
binding_func_gen(gemv_fp32x4, 4, float);

// binding
#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(gemv);
    torch_pybinding_func(gemv_fp32x4);
}
