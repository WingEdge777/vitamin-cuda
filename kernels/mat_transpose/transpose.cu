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

// naive coalesced read transpose
__global__ void transpose_coalesced_read_kernel(float *a, float *b, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        b[x * height + y] = a[y * width + x];
    }
}

// naive coalesced write transpose
__global__ void transpose_coalesced_write_kernel(float *a, float *b, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < height && y < width) {
        b[y * height + x] = a[x * width + y];
    }
}

// transpose with smem
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
// Fast Path
#pragma unroll
        for (int j = 0; j < tiling_size; j += tiling_row) {
            tile[ty + j][tx] = a[(y + j) * width + x];
        }
    } else {
// Slow Path
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

// bank conflict free
__global__ void transpose_smem_bcf_kernel(float *a, float *b, int width, int height) {
    __shared__ float tile[tiling_size][tiling_size + 1];
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

// smem bcf + float4 r/w
__global__ void transpose_smem_packed_bcf_kernel(float *a, float *b, int width, int height) {
    __shared__ float tile[tiling_size][tiling_size + 1];
    int tid = threadIdx.x + threadIdx.y * blockDim.x; // [0, 256]

    int sx = tid % 8;
    int sy = tid / 8;

    int a_x = blockIdx.x * tiling_size + sx * 4;
    int a_y = blockIdx.y * tiling_size + sy;
    if (a_x < width && a_y < height) {
        float4 va = LDST128BITS(a[a_y * width + a_x]);
        tile[sy][sx * 4 + 0] = va.x;
        tile[sy][sx * 4 + 1] = va.y;
        tile[sy][sx * 4 + 2] = va.z;
        tile[sy][sx * 4 + 3] = va.w;
    }

    __syncthreads();

    int b_x = blockIdx.y * tiling_size + sx * 4;
    int b_y = blockIdx.x * tiling_size + sy;
    if (b_x < height && b_y < width) {
        float4 vb;
        vb.x = tile[sx * 4 + 0][sy];
        vb.y = tile[sx * 4 + 1][sy];
        vb.z = tile[sx * 4 + 2][sy];
        vb.w = tile[sx * 4 + 3][sy];
        LDST128BITS(b[b_y * height + b_x]) = vb;
    }
}

// smem swizzle bcf + float4 r/w
__global__ void transpose_smem_swizzled_packed_kernel(float *a, float *b, int width, int height) {
    __shared__ float tile[tiling_size][tiling_size];
    int tid = threadIdx.x + threadIdx.y * blockDim.x; // [0, 256]

    int sx = tid % 8;
    int sy = tid / 8;
    int a_x = blockIdx.x * tiling_size + sx * 4;
    int a_y = blockIdx.y * tiling_size + sy;
    if (a_x < width && a_y < height) {
        float4 va = LDST128BITS(a[a_y * width + a_x]);

        tile[sy][(sx * 4 + 0) ^ sy] = va.x;
        tile[sy][(sx * 4 + 1) ^ sy] = va.y;
        tile[sy][(sx * 4 + 2) ^ sy] = va.z;
        tile[sy][(sx * 4 + 3) ^ sy] = va.w;
    }

    __syncthreads();

    int b_x = blockIdx.y * tiling_size + sx * 4;
    int b_y = blockIdx.x * tiling_size + sy;
    if (b_x < height && b_y < width) {
        float4 vb;

        vb.x = tile[sx * 4 + 0][sy ^ (sx * 4 + 0)];
        vb.y = tile[sx * 4 + 1][sy ^ (sx * 4 + 1)];
        vb.z = tile[sx * 4 + 2][sy ^ (sx * 4 + 2)];
        vb.w = tile[sx * 4 + 3][sy ^ (sx * 4 + 3)];

        LDST128BITS(b[b_y * height + b_x]) = vb;
    }
}

#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

void transpose_coalesced_read(torch::Tensor a, torch::Tensor b) {
    CHECK_T(a);
    CHECK_T(b);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int height = a.size(0);
    const int width = a.size(1);
    const dim3 threads_per_block(naive_tiling_size, naive_tiling_size);
    const dim3 blocks_per_grid((width + naive_tiling_size - 1) / naive_tiling_size,
                               (height + naive_tiling_size - 1) / naive_tiling_size);
    transpose_coalesced_read_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), width, height);
}
void transpose_coalesced_write(torch::Tensor a, torch::Tensor b) {
    CHECK_T(a);
    CHECK_T(b);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int height = a.size(0);
    const int width = a.size(1);
    const dim3 threads_per_block(naive_tiling_size, naive_tiling_size);
    const dim3 blocks_per_grid((height + naive_tiling_size - 1) / naive_tiling_size,
                               (width + naive_tiling_size - 1) / naive_tiling_size);
    transpose_coalesced_write_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), width, height);
}

#define binding_func_gen(name, num, element_dtype)                                                                     \
    void name(torch::Tensor a, torch::Tensor b) {                                                                      \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        const int height = a.size(0);                                                                                  \
        const int width = a.size(1);                                                                                   \
        const dim3 threads_per_block(tiling_size, tiling_row);                                                         \
        const dim3 blocks_per_grid((width + tiling_size - 1) / tiling_size, (height + tiling_size - 1) / tiling_size); \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
                                                                                                                       \
        name##_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(                                              \
            a.data_ptr<float>(), b.data_ptr<float>(), width, height);                                                  \
    }

binding_func_gen(transpose_smem, 1, float);
binding_func_gen(transpose_smem_bcf, 1, float);
binding_func_gen(transpose_smem_packed_bcf, 4, float);
binding_func_gen(transpose_smem_swizzled_packed, 4, float);

// binding
#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(transpose_coalesced_read);
    torch_pybinding_func(transpose_coalesced_write);
    torch_pybinding_func(transpose_smem);
    torch_pybinding_func(transpose_smem_bcf);
    torch_pybinding_func(transpose_smem_packed_bcf);
    torch_pybinding_func(transpose_smem_swizzled_packed);
}
