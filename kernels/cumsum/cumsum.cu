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
template <const int BLOCK_SIZE = 256, const int CHUNK_SIZE = 256>
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
            tmp_v = _warp_cum_sum<num_warp>(lane_id, tmp_v);
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

template <const int warp_size = WARP_SIZE>
__device__ __forceinline__ void _warp_cum_sum(int lane_id, pack128 &val) {
#pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float other_val = __shfl_up_sync(0xffffffff, val.f[3], offset);
        if (lane_id >= offset) {
#pragma unroll
            for (int k = 0; k < 4; k++)
                val.f[k] += other_val;
        }
    }
}

template <const int BLOCK_SIZE = 256, const int CHUNK_SIZE = 1024>
__global__ void cumsum_fp32x4_kernel(float *a, float *b, int n) {
    int tid = threadIdx.x;
    int row = blockIdx.x;

    const int num_warp = BLOCK_SIZE / 32;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    __shared__ float smem[num_warp];
    __shared__ float block_delta;

    float delta = 0.f;
    int row_offset = row * n;
    for (int i = 0; i < n; i += CHUNK_SIZE) {
        int col = i + tid * 4;
        pack128 f4;
        f4.f4 = (col < n) ? FLOAT4(a[row_offset + col]) : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int k = 1; k < 4; k++)
            f4.f[k] += f4.f[k - 1];
        _warp_cum_sum<32>(lane_id, f4);

        if (lane_id == 32 - 1)
            smem[warp_id] = f4.f[3];
        __syncthreads();

        if (warp_id == 0) {
            float tmp_v = tid < num_warp ? smem[tid] : 0.f;
            tmp_v = _warp_cum_sum<num_warp>(lane_id, tmp_v);
            if (tid < num_warp)
                smem[tid] = tmp_v;
        }
        __syncthreads();

        float prefix = delta + (warp_id > 0 ? smem[warp_id - 1] : 0.f);
#pragma unroll
        for (int k = 0; k < 4; k++)
            f4.f[k] += prefix;

        if (col < n) {
            FLOAT4(b[row_offset + col]) = f4.f4;
        }

        if (tid == BLOCK_SIZE - 1) {
            block_delta = f4.f[3];
        }

        __syncthreads();

        delta = block_delta;
    }
}

// Decoupled Look-back (Merrill & Garland).
// Pack {status:high32, float_bits:low32} so status and payload are one atomic word.
enum ScanTileStatus : unsigned int {
    TILE_INVALID = 0,  // must be 0: workspace is zero-filled
    TILE_PARTIAL = 1,  // aggregate of this tile only
    TILE_INCLUSIVE = 2 // inclusive prefix through end of this tile
};

__device__ __forceinline__ unsigned long long pack_tile_state(ScanTileStatus status, float value) {
    return (static_cast<unsigned long long>(status) << 32) | static_cast<unsigned long long>(__float_as_uint(value));
}

__device__ __forceinline__ void unpack_tile_state(unsigned long long packed, ScanTileStatus &status, float &value) {
    status = static_cast<ScanTileStatus>(static_cast<unsigned int>(packed >> 32));
    value = __uint_as_float(static_cast<unsigned int>(packed));
}

__device__ __forceinline__ void store_tile_state(unsigned long long *addr, ScanTileStatus status, float value) {
    atomicExch(addr, pack_tile_state(status, value));
    __threadfence();
}

__device__ __forceinline__ void load_tile_state(unsigned long long *addr, ScanTileStatus &status, float &value) {
    unpack_tile_state(atomicOr(addr, 0ull), status, value);
}

template <const int BLOCK_SIZE = 256, const int CHUNK_SIZE = 1024>
__global__ void
cumsum_fp32x4_multi_cta_scan_kernel(float *a, float *b, unsigned long long *tile_state, int n, int num_chunks) {
    int tid = threadIdx.x;
    int chunk_id = blockIdx.x;
    int row = blockIdx.y;

    const int num_warp = BLOCK_SIZE / 32;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    __shared__ float smem[num_warp];
    __shared__ float exclusive_prefix_smem;

    int row_offset = row * n;
    int col = chunk_id * CHUNK_SIZE + tid * 4;

    pack128 f4;
    f4.f4 = (col < n) ? FLOAT4(a[row_offset + col]) : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int k = 1; k < 4; k++)
        f4.f[k] += f4.f[k - 1];
    _warp_cum_sum<32>(lane_id, f4);

    if (lane_id == 32 - 1)
        smem[warp_id] = f4.f[3];
    __syncthreads();

    if (warp_id == 0) {
        float tmp_v = tid < num_warp ? smem[tid] : 0.f;
        tmp_v = _warp_cum_sum<num_warp>(lane_id, tmp_v);
        if (tid < num_warp)
            smem[tid] = tmp_v;
    }
    __syncthreads();

    float local_prefix = (warp_id > 0 ? smem[warp_id - 1] : 0.f);
#pragma unroll
    for (int k = 0; k < 4; k++)
        f4.f[k] += local_prefix;

    // Block aggregate = last thread's inclusive sum (tail padding is zero).
    if (tid == BLOCK_SIZE - 1) {
        smem[0] = f4.f[3];
    }
    __syncthreads();
    float block_sum = smem[0];

    unsigned long long *row_tiles = tile_state + static_cast<long long>(row) * num_chunks;
    float exclusive_prefix = 0.f;

    // tid0 decoupled look-back: publish PARTIAL, scan predecessors, upgrade to INCLUSIVE.
    if (tid == 0) {
        if (chunk_id == 0) {
            store_tile_state(row_tiles + 0, TILE_INCLUSIVE, block_sum);
            exclusive_prefix = 0.f;
        } else {
            store_tile_state(row_tiles + chunk_id, TILE_PARTIAL, block_sum);

            int lookback = chunk_id - 1;
            while (true) {
                ScanTileStatus status;
                float value;
                load_tile_state(row_tiles + lookback, status, value);
                if (status == TILE_INCLUSIVE) {
                    exclusive_prefix += value;
                    break;
                }
                if (status == TILE_PARTIAL) {
                    exclusive_prefix += value;
                    --lookback; // tile 0 never publishes PARTIAL, so lookback stays >= 0
                }
                // TILE_INVALID: spin on the same predecessor
            }

            store_tile_state(row_tiles + chunk_id, TILE_INCLUSIVE, exclusive_prefix + block_sum);
        }
        exclusive_prefix_smem = exclusive_prefix;
    }
    __syncthreads();

    exclusive_prefix = exclusive_prefix_smem;
#pragma unroll
    for (int k = 0; k < 4; k++)
        f4.f[k] += exclusive_prefix;

    if (col < n) {
        FLOAT4(b[row_offset + col]) = f4.f4;
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

binding_func_gen(cumsum_fp32, 1, float);
binding_func_gen(cumsum_fp32x4, 4, float);

void cumsum_fp32x4_multi_cta_scan(torch::Tensor a, torch::Tensor b) {
    CHECK_T(a);
    CHECK_T(b);
    const int bs = a.size(0);
    const int lda = a.size(1);
    constexpr int threads_per_block = 256;
    constexpr int vec = 4;
    constexpr int chunk_size = threads_per_block * vec;
    const int num_chunks = (lda + chunk_size - 1) / chunk_size;
    auto stream = at::cuda::getCurrentCUDAStream();

    auto options = torch::TensorOptions().dtype(torch::kLong).device(a.device());
    auto tile_state = torch::empty({bs, num_chunks}, options);
    // Explicit memset on the launch stream + record_stream so raw <<<>>> cannot race the allocator.
    tile_state.record_stream(stream);
    TORCH_CHECK(cudaMemsetAsync(tile_state.data_ptr(), 0, tile_state.nbytes(), stream.stream()) == cudaSuccess);

    dim3 grid(num_chunks, bs);
    cumsum_fp32x4_multi_cta_scan_kernel<threads_per_block, chunk_size>
        <<<grid, threads_per_block, 0, stream.stream()>>>(reinterpret_cast<float *>(a.data_ptr()),
                                                          reinterpret_cast<float *>(b.data_ptr()),
                                                          reinterpret_cast<unsigned long long *>(tile_state.data_ptr()),
                                                          lda,
                                                          num_chunks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(cumsum_fp32);
    torch_pybinding_func(cumsum_fp32x4);
    torch_pybinding_func(cumsum_fp32x4_multi_cta_scan);
}
