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
const int naive_tiling_size = 16;

const int tiling_size = 32;
const int tiling_row = 8;

// coalesced transpose
__global__ void transpose_coalesced_read_kernel(float *a, float *b, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        b[x * height + y] = a[y * width + x];
    }
}

__global__ void transpose_coalesced_write_kernel(float *a, float *b, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        b[y * height + x] = a[x * width + y];
    }
}

// transpose
__global__ void transpose_smem_kernel(float *a, float *b, int width, int height) {
    __shared__ float tile[tiling_size][tiling_size];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int x = bx * tiling_size + tx;
    int y = by * tiling_size + ty;

    bool x_full = (bx + 1) * tiling_size <= width;
    bool y_full = (by + 1) * tiling_size <= height;

    if (x_full && y_full) {
#pragma unroll
        for (int j = 0; j < tiling_size; j += tiling_row) {
            tile[ty + j][tx] = a[(y + j) * width + x];
        }
    } else {
#pragma unroll
        for (int j = 0; j < tiling_size; j += tiling_row) {
            if (x < width && (y + j) < height) {
                tile[ty + j][tx] = a[(y + j) * width + x];
            }
        }
    }

    __syncthreads();
    x = by * tiling_size + tx;
    y = bx * tiling_size + ty;

    bool write_x_full = (by + 1) * tiling_size <= height;
    bool write_y_full = (bx + 1) * tiling_size <= width;

    if (write_x_full && write_y_full) {
// Fast Path
#pragma unroll
        for (int j = 0; j < tiling_size; j += tiling_row) {
            b[(y + j) * height + x] = tile[tx][ty + j];
        }
    } else {
// Slow Path
#pragma unroll
        for (int j = 0; j < tiling_size; j += tiling_row) {
            if (x < height && (y + j) < width) {
                b[(y + j) * height + x] = tile[tx][ty + j];
            }
        }
    }
}

__global__ void transpose_fp32x4_kernel(float *a, float *b, int width, int height) {}

#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")
#define binding_coalesced_gen(name, num, element_dtype)                                                                \
    void name(torch::Tensor a, torch::Tensor b) {                                                                      \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        const int height = a.size(0);                                                                                  \
        const int width = a.size(1);                                                                                   \
        const dim3 threads_per_block(naive_tiling_size, naive_tiling_size);                                            \
        const dim3 blocks_per_grid((width + naive_tiling_size - 1) / naive_tiling_size,                                \
                                   (height + naive_tiling_size - 1) / naive_tiling_size);                              \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
                                                                                                                       \
        name##_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(                                              \
            reinterpret_cast<float *>(a.data_ptr()), reinterpret_cast<float *>(b.data_ptr()), width, height);          \
    }

#define binding_func_gen(name, num, element_dtype)                                                                     \
    void name(torch::Tensor a, torch::Tensor b) {                                                                      \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        const int height = a.size(0);                                                                                  \
        const int width = a.size(1);                                                                                   \
        const dim3 threads_per_block(tiling_size, tiling_row);                                                         \
        const dim3 blocks_per_grid((width + tiling_size - 1) / tiling_size, (height + tiling_row - 1) / tiling_row);   \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
                                                                                                                       \
        name##_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(                                              \
            reinterpret_cast<float *>(a.data_ptr()), reinterpret_cast<float *>(b.data_ptr()), width, height);          \
    }
binding_coalesced_gen(transpose_coalesced_read, 1, float);
binding_coalesced_gen(transpose_coalesced_write, 1, float);
binding_func_gen(transpose_smem, 1, float);
binding_func_gen(transpose_smem_bcf, 1, float);
binding_func_gen(transpose_smem_bcf_packed, 1, float);

// binding
#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(transpose_coalesced_read);
    torch_pybinding_func(transpose_coalesced_write);
    torch_pybinding_func(transpose_smem);
    torch_pybinding_func(transpose_smem_bcf);
    torch_pybinding_func(transpose_smem_bcf_packed);
}
