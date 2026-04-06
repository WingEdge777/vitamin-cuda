# [CUDA in Practice] Safe Online Softmax — A Must-Know for Interviews: Arbitrary hidden_size, One/Two Pass, Trade-offs, Split-K
>
>Note: Text translated by AI. Code crafted by human.
>

# 0. Preface — Background

Softmax is a ubiquitous operator in deep learning, used across virtually every ML domain for confidence/weight/probability output predictions. Without softmax, a model has no output (embeddings aside).

> I have a personal fondness for softmax — during my campus recruitment, the final round manager interview ended with me whiteboarding a softmax forward pass in numpy.

In today's large-scale classification models and LLMs, softmax is being pushed to its limits from every angle. Much like GEMM, it demands tuning across a wide variety of scenarios.

If I were the interviewer, I'd absolutely ask candidates about softmax optimization across different scenarios. This post is a collection of various softmax implementations and the considerations behind them.

> There is no best kernel, only the most suitable kernel.
> ---------------------------------- altum sonatur (throw in some Latin and it instantly sounds classy)

Note: All kernels listed in this article are for demonstration purposes only and have not been hand-tuned to the extreme. All input data is assumed to be 128-byte aligned.

This article presents two categories of kernel implementations, outlined as follows:

- One-pass
  - softmax fp16x8 version (fp16 vectorized, pure register cache, packed r/w)
  - softmax medium fp16 version (fp16 vectorized, medium register+smem cache usage, packed r/w)
  - softmax extreme fp16 version (fp16 vectorized, maximum register+smem cache usage, packed r/w)
- Two-pass
  - softmax arbitrary fp16 version (fp16 vectorized, packed r/w)
  - softmax split-k fp16 version (fp16 vectorized, packed r/w)

Here "pass" refers to the number of traversals over the input.

- One-pass kernels: The core idea is to cache the input in registers and shared memory. After reducing max and exp_sum, the output is computed by reading input directly from the register/smem cache.
- Two-pass kernels: The first is a naive implementation with no input caching — read as much as you compute. The second follows the now-popular split-k approach — when batch size is small but the dimension is very large, the dimension is partitioned into chunks. Each chunk undergoes a block-level reduction, and then a second reduction merges results across all chunks.

Of course, all implementations above are built on the "safe" premise. Whether to go online or not depends on the scale and the specific phase of the kernel.

# 1. Safe Online Softmax

By now everyone has likely heard of safe online softmax (the foundation of FlashAttention), so here are three numpy versions for quick comparison with comments:

Softmax:

```python
import numpy as np

def softmax(x):
    """
    Basic version, prone to numerical overflow.
    Formula: exp(x) / sum(exp(x))
    """
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

Safe softmax:

```python
def safe_softmax(x):
    """
    Safe version, leverages translation invariance to prevent overflow.
    Formula: exp(x - max) / sum(exp(x - max))
    """
    # Pass 1: Find the global maximum
    x_max = np.max(x, axis=-1, keepdims=True)

    # Pass 2: Subtract the max, compute exp, and normalize
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

Safe online softmax:

```python
def safe_online_softmax(x, block_size=256):
    """
    Online version
    """
    batch_size, hidden_size = x.shape

    # Global max (m) and sum (d)
    m = np.full((batch_size, 1), -np.inf)
    d = np.zeros((batch_size, 1))

    # Pass 1: Compute global max and sum online, chunk by chunk
    for i in range(0, hidden_size, block_size):
        chunk = x[:, i:i+block_size]

        # Find the local maximum of the current chunk
        m_local = np.max(chunk, axis=-1, keepdims=True)

        # Core online update logic
        m_new = np.maximum(m, m_local)

        # Compute the local sum of the current chunk, based on m_new
        d_local = np.sum(np.exp(chunk - m_new), axis=-1, keepdims=True)

        # Rescale the historical sum (d) and add the current local sum
        d = d * np.exp(m - m_new) + d_local

        # Update the global max
        m = m_new

    # Pass 2: Write back results using the computed global m and d
    out = np.zeros_like(x)
    for i in range(0, hidden_size, block_size):
        chunk = x[:, i:i+block_size]
        out[:, i:i+block_size] = np.exp(chunk - m) / d

    return out
```

# 2. One-Pass Kernels

## 2.1 Pure Register Cache

We assume the input is 2D with shape `[batch_size, hidden_size]`, and the naive idea is to have one block process one batch row.

First, I chose a block size of 256. Don't ask why — my GPU works well with 128 and 256, and I like 256.

Then, given my total register file (65536 registers), I calculated the maximum number of registers available per thread (256). To keep occupancy from being too low (we don't want an SM with only one or even zero active blocks, just sitting there waiting — though in extreme cases this is intentional and called a persistent kernel), I chose 64 registers.

This choice isn't great, because the kernel definitely has additional register usage for pointers, offsets, temporary computation buffers, etc. — probably pushing it to 80–100 registers total. So only 2–3 blocks can be active. But since this is just for demonstration and not chasing peak performance, and 64 32-bit registers as cache can cover all cases up to hidden_size <= 8192, I went with 64.

Here's the code:

```cpp

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

const int WARP_SIZE = 32;

struct __align__(8) MD {
    float m;
    float d;
};

union pack128 {
    float4 f4;
    half2 h2[4];
};

template <const int warp_size = WARP_SIZE>
__device__ __forceinline__ MD _warp_online_softmax_reduce(MD val) {
#pragma unroll
    for (int mask = warp_size >> 1; mask > 0; mask >>= 1) {
        float other_m = __shfl_xor_sync(0xffffffff, val.m, mask);
        float other_d = __shfl_xor_sync(0xffffffff, val.d, mask);

        float new_m = fmaxf(val.m, other_m);
        val.d = val.d * __expf(val.m - new_m) + other_d * __expf(other_m - new_m);
        val.m = new_m;
    }
    return val;
}

template <const int BLOCK_SIZE = 256>
__device__ __forceinline__ MD block_online_softmax_reduce(MD val) {
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    val = _warp_online_softmax_reduce<WARP_SIZE>(val);

    __shared__ MD sdata[NUM_WARPS];
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (lane_id == 0)
        sdata[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = lane_id < NUM_WARPS ? sdata[lane_id] : MD{-FLT_MAX, 0.f};
        val = _warp_online_softmax_reduce<NUM_WARPS>(val);
        if (lane_id == 0)
            sdata[0] = val;
    }
    __syncthreads();
    val = sdata[0];

    return val;
}

// pure register cache, hidden_size <= 8192
template <const int BLOCK_SIZE = 256>
__global__ void softmax_fp16x8_packed_kernel(half *a, half *b, int hidden_size) {
    int row_offset = blockIdx.x * hidden_size;
    int tid = threadIdx.x;

    pack128 pack[16];
    MD val{-FLT_MAX, 0.f};
    constexpr int MAX_VECS = 16;

#pragma unroll
    for (int i = 0; i < MAX_VECS; i++) {
        int offset = (tid + i * BLOCK_SIZE) * 8;
        if (offset < hidden_size) {
            pack[i].f4 = LDST128BITS(a[row_offset + offset]);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float2 f2 = __half22float2(pack[i].h2[j]);
                val.m = fmaxf(val.m, fmaxf(f2.x, f2.y));
            }
        }
    }

#pragma unroll
    for (int i = 0; i < MAX_VECS; i++) {
        int offset = (tid + i * BLOCK_SIZE) * 8;
        if (offset < hidden_size) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float2 f2 = __half22float2(pack[i].h2[j]);
                val.d += __expf(f2.x - val.m) + __expf(f2.y - val.m);
            }
        }
    }

    val = block_online_softmax_reduce<BLOCK_SIZE>(val);

#pragma unroll
    for (int i = 0; i < MAX_VECS; i++) {
        int offset = (tid + i * BLOCK_SIZE) * 8;
        if (offset < hidden_size) {
            pack128 out;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float2 f2 = __half22float2(pack[i].h2[j]);
                f2.x = __expf(f2.x - val.m) / val.d;
                f2.y = __expf(f2.y - val.m) / val.d;
                out.h2[j] = __float22half2_rn(f2);
            }
            LDST128BITS(b[row_offset + offset]) = out.f4;
        }
    }
}
```

When reading the input, we directly compute the per-thread local maximum in the same loop. The online softmax idea (repeated rescaling) is only applied during the warp reduce phase.

### 2.1.1 Warp Reduce

For warp reduce, we use the `__shfl_xor_sync` warp-level primitive, which performs synchronous register-variable exchange within a warp. For example:

```cpp
for (int mask = warp_size >> 1; mask > 0; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
}
```

The logic of `__shfl_xor_sync` is that thread `id` can access the `val` of thread `id ^ mask`.
Therefore, this loop executes five iterations:

- First iteration: threads 0–15 and 16–31 exchange values pairwise and take the maximum.
- Second iteration: threads 0–7 and 8–15, threads 16–23 and 24–31.
- And so on...

This process is known as the classic butterfly reduction. For more details, see the official documentation on `__shfl_xor_sync`: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=__shfl_xor_sync#warp-shuffle-functions>

Or search for other blogs to learn the details.

## 2.2 Register + Shared Memory Cache

### 2.2.1 Medium Kernel

This kernel aims to utilize both registers and shared memory. Given my GPU's constraints, the maximum shared memory per block is 48KB, so it can cache at most 48×1024 / 2 / 256 = 96 half values per thread. However, the block reduce also needs a small amount of shared memory, so I can't max it out. Since this is the "medium" version intended for demonstration, I chose the same size as the register cache above — 64 half values per thread. This way, each block uses 64×256×2 = 32KB. My SM has a total of 100KB shared memory, so 3 active blocks can fit, which is close to the register-limited case above.

With this configuration, the maximum supported hidden_size is 32768.

OK, after analyzing the hardware, I should be showing the code, but to reduce code volume I merged the medium and extreme versions into a single kernel, differentiated by template parameters. See the next section~

### 2.2.2 Extreme Kernel

"Extreme" means handling the largest possible hidden_size under the one-pass constraint, so we push register and shared memory usage to the max.

- Shared memory first: This is straightforward to calculate. Total 100KB, minus a tiny amount for warp reduce usage. After verification, I rounded to 96KB, which can cache 96×1024 / 256 / 2 = 192 half values per thread (24 float4 vectors).
  - Note: Since we need to exceed the per-block 48KB limit, we must use dynamic shared memory and set the shared memory size via the CUDA API `cudaFuncSetAttribute`.
- Registers: Although the per-thread maximum is 256 registers as calculated earlier, our code uses extensive unrolling with multiple type conversions, max/exp computations, etc., making register pressure very high. After verification, I settled on 128 registers, i.e., 256 half values (32 float4 vectors).
  - Note: Actual usage measured at 250 registers.

With this configuration, the maximum supported hidden_size is 114688.

OK, now for the code:

```cpp
template <const int BLOCK_SIZE, const int REG_VECS, const int SMEM_VECS, const int MIN_BLOCKS_PER_SM>
__global__ __launch_bounds__(BLOCK_SIZE,
                             MIN_BLOCKS_PER_SM) void softmax_onepass_kernel(half *a, half *b, int hidden_size) {
    int row_offset = blockIdx.x * hidden_size;
    int tid = threadIdx.x;

    pack128 reg_pack[REG_VECS];
    extern __shared__ float4 smem_pack_flat[];
    float4(*smem_pack)[BLOCK_SIZE] = reinterpret_cast<float4(*)[BLOCK_SIZE]>(smem_pack_flat);

    float local_m = -FLT_MAX;

#pragma unroll
    for (int i = 0; i < REG_VECS; i++) {
        int col_idx = (tid + i * BLOCK_SIZE) * 8;
        if (col_idx < hidden_size) {
            reg_pack[i].f4 = LDST128BITS(a[row_offset + col_idx]);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float2 f2 = __half22float2(reg_pack[i].h2[j]);
                local_m = fmaxf(local_m, fmaxf(f2.x, f2.y));
            }
        }
    }

#pragma unroll
    for (int i = 0; i < SMEM_VECS; i++) {
        int global_i = i + REG_VECS;
        int col_idx = (tid + global_i * BLOCK_SIZE) * 8;
        if (col_idx < hidden_size) {
            pack128 tmp;
            tmp.f4 = LDST128BITS(a[row_offset + col_idx]);
            smem_pack[i][tid] = tmp.f4;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float2 f2 = __half22float2(tmp.h2[j]);
                local_m = fmaxf(local_m, fmaxf(f2.x, f2.y));
            }
        }
    }

    float local_d = 0.f;
#pragma unroll
    for (int i = 0; i < REG_VECS; i++) {
        int col_idx = (tid + i * BLOCK_SIZE) * 8;
        if (col_idx < hidden_size) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float2 f2 = __half22float2(reg_pack[i].h2[j]);
                local_d += __expf(f2.x - local_m) + __expf(f2.y - local_m);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < SMEM_VECS; i++) {
        int global_i = i + REG_VECS;
        int col_idx = (tid + global_i * BLOCK_SIZE) * 8;
        if (col_idx < hidden_size) {
            pack128 tmp;
            tmp.f4 = smem_pack[i][tid];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float2 f2 = __half22float2(tmp.h2[j]);
                local_d += __expf(f2.x - local_m) + __expf(f2.y - local_m);
            }
        }
    }

    MD val{local_m, local_d};
    val = block_online_softmax_reduce<BLOCK_SIZE>(val);

#pragma unroll
    for (int i = 0; i < REG_VECS; i++) {
        int col_idx = (tid + i * BLOCK_SIZE) * 8;
        if (col_idx < hidden_size) {
            pack128 out;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float2 f2 = __half22float2(reg_pack[i].h2[j]);
                f2.x = __expf(f2.x - val.m) / val.d;
                f2.y = __expf(f2.y - val.m) / val.d;
                out.h2[j] = __float22half2_rn(f2);
            }
            LDST128BITS(b[row_offset + col_idx]) = out.f4;
        }
    }
#pragma unroll
    for (int i = 0; i < SMEM_VECS; i++) {
        int global_i = i + REG_VECS;
        int col_idx = (tid + global_i * BLOCK_SIZE) * 8;
        if (col_idx < hidden_size) {
            pack128 tmp;
            tmp.f4 = smem_pack[i][tid];
            pack128 out;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float2 f2 = __half22float2(tmp.h2[j]);
                f2.x = __expf(f2.x - val.m) / val.d;
                f2.y = __expf(f2.y - val.m) / val.d;
                out.h2[j] = __float22half2_rn(f2);
            }
            LDST128BITS(b[row_offset + col_idx]) = out.f4;
        }
    }
}

#define binding_single_launch_gen(name, smem_bytes, ...)                                                               \
    void name(torch::Tensor a, torch::Tensor b) {                                                                      \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        const int bs = a.size(0);                                                                                      \
        const int lda = a.size(1);                                                                                     \
        const int threads_per_block = 256;                                                                             \
        const int blocks_per_grid = bs;                                                                                \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
        if (smem_bytes > 49152) {                                                                                      \
            cudaError_t err =                                                                                          \
                cudaFuncSetAttribute(__VA_ARGS__, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);            \
            TORCH_CHECK(err == cudaSuccess, "Failed to unlock Shared Memory limit!");                                  \
        }                                                                                                              \
        __VA_ARGS__<<<blocks_per_grid, threads_per_block, smem_bytes, stream>>>(                                       \
            reinterpret_cast<half *>(a.data_ptr()), reinterpret_cast<half *>(b.data_ptr()), lda);                      \
    }

binding_single_launch_gen(softmax_medium, 32768, softmax_onepass_kernel<256, 8, 8, 2>);
binding_single_launch_gen(softmax_extreme, 98304, softmax_onepass_kernel<256, 32, 24, 1>);
```

The code looks a bit long, but it's essentially applying the same processing to both registers and shared memory. The related logic could be refactored into functions, but I was too lazy to do it — apologies~

# 3. Two-Pass

## 3.1 Vanilla Implementation Supporting Arbitrary hidden_size

Since it's two-pass, it should naturally support arbitrary sizes (as long as you don't run out of VRAM). Here's a naive implementation: vectorized loads, then online-style updates of max and sum.

Code:

```cpp
// two pass : Vanilla
template <const int BLOCK_SIZE = 256>
__global__ void softmax_arbitrary_kernel(half *a, half *b, int hidden_size) {
    int row_offset = blockIdx.x * hidden_size;
    int tid = threadIdx.x;

    MD val{-FLT_MAX, 0.f};

    for (int i = 0; (tid + i * BLOCK_SIZE) * 8 < hidden_size; i++) {
        pack128 tmp;
        tmp.f4 = LDST128BITS(a[row_offset + (tid + i * BLOCK_SIZE) * 8]);

        float local_m = -FLT_MAX;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 f2 = __half22float2(tmp.h2[j]);
            local_m = fmaxf(local_m, fmaxf(f2.x, f2.y));
        }

        float new_m = fmaxf(val.m, local_m);
        float scale = __expf(val.m - new_m);
        val.d *= scale;

#pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 f2 = __half22float2(tmp.h2[j]);
            val.d += __expf(f2.x - new_m) + __expf(f2.y - new_m);
        }
        val.m = new_m;
    }

    val = block_online_softmax_reduce<BLOCK_SIZE>(val);

    for (int i = 0; (tid + i * BLOCK_SIZE) * 8 < hidden_size; i++) {
        pack128 tmp;
        tmp.f4 = LDST128BITS(a[row_offset + (tid + i * BLOCK_SIZE) * 8]);
        pack128 out;

#pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 f2 = __half22float2(tmp.h2[j]);
            f2.x = __expf(f2.x - val.m) / val.d;
            f2.y = __expf(f2.y - val.m) / val.d;
            out.h2[j] = __float22half2_rn(f2);
        }
        LDST128BITS(b[row_offset + (tid + i * BLOCK_SIZE) * 8]) = out.f4;
    }
}
```

As you can see, the code is quite clean: vectorized reads, online-style per-thread max updates, a block-level reduce, and then a second read from `a` when writing back to `b`.

It looks naive, and it is. But in practice, the performance is surprisingly good — because when hidden_size isn't too large, after reading `a` once, the data is cached in L2. The second pass essentially reads from L2.

## 3.2 Split-K Kernel

Split-K is designed to accelerate computation when hidden_size is very large but batch size is small. The core idea is to partition along the hidden_size dimension. Each block computes the local max and sum within its chunk and writes the results to global memory.

Then a second kernel is launched to read and merge all chunk results with another reduction, re-read the input, and compute the final output.

```cpp
template <const int BLOCK_SIZE>
__device__ __forceinline__ MD block_online_softmax_reduce_no_broadcast(MD val) {
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    val = _warp_online_softmax_reduce<WARP_SIZE>(val);

    __shared__ MD sdata[NUM_WARPS];
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (lane_id == 0)
        sdata[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = lane_id < NUM_WARPS ? sdata[lane_id] : MD{-FLT_MAX, 0.f};
        val = _warp_online_softmax_reduce<NUM_WARPS>(val);
    }

    return val;
}

// split-k pass 1
template <const int BLOCK_SIZE = 256, const int VECS_PER_THREAD = 16>
__global__ void softmax_grid_pass1(half *a, float *ws_m, float *ws_d, int hidden_size) {
    int row = blockIdx.y;
    int chunk_id = blockIdx.x;
    int tid = threadIdx.x;

    int chunk_offset = chunk_id * (BLOCK_SIZE * VECS_PER_THREAD * 8);
    int col_offset = chunk_offset + tid * 8;

    MD val{-FLT_MAX, 0.f};

    pack128 cache[VECS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < VECS_PER_THREAD; i++) {
        int col_idx = col_offset + i * BLOCK_SIZE * 8;
        if (col_idx < hidden_size) {
            cache[i].f4 = LDST128BITS(a[row * hidden_size + col_idx]);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float2 f2 = __half22float2(cache[i].h2[j]);
                val.m = fmaxf(val.m, fmaxf(f2.x, f2.y));
            }
        }
    }

#pragma unroll
    for (int i = 0; i < VECS_PER_THREAD; i++) {
        int col_idx = col_offset + i * BLOCK_SIZE * 8;
        if (col_idx < hidden_size) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float2 f2 = __half22float2(cache[i].h2[j]);
                val.d += __expf(f2.x - val.m) + __expf(f2.y - val.m);
            }
        }
    }

    val = block_online_softmax_reduce_no_broadcast<BLOCK_SIZE>(val);

    if (tid == 0) {
        ws_m[row * gridDim.x + chunk_id] = val.m;
        ws_d[row * gridDim.x + chunk_id] = val.d;
    }
}

// splitk pass 2
template <const int BLOCK_SIZE = 256, const int VECS_PER_THREAD = 16>
__global__ void softmax_grid_pass2(half *a, half *b, float *ws_m, float *ws_d, int hidden_size) {
    int row = blockIdx.y;
    int chunk_id = blockIdx.x;
    int tid = threadIdx.x;
    int blocks_per_row = gridDim.x;

    MD global_val{-FLT_MAX, 0.f};
    for (int i = tid; i < blocks_per_row; i += BLOCK_SIZE) {
        float other_m = ws_m[row * blocks_per_row + i];
        float other_d = ws_d[row * blocks_per_row + i];
        float new_m = fmaxf(global_val.m, other_m);
        global_val.d = global_val.d * __expf(global_val.m - new_m) + other_d * __expf(other_m - new_m);
        global_val.m = new_m;
    }

    global_val = block_online_softmax_reduce<BLOCK_SIZE>(global_val);

    int chunk_offset = chunk_id * (BLOCK_SIZE * VECS_PER_THREAD * 8);
    int col_offset = chunk_offset + tid * 8;

#pragma unroll
    for (int i = 0; i < VECS_PER_THREAD; i++) {
        int col_idx = col_offset + i * BLOCK_SIZE * 8;
        if (col_idx < hidden_size) {
            pack128 tmp;
            tmp.f4 = LDST128BITS(a[row * hidden_size + col_idx]);
            pack128 out;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float2 f2 = __half22float2(tmp.h2[j]);
                f2.x = __expf(f2.x - global_val.m) / global_val.d;
                f2.y = __expf(f2.y - global_val.m) / global_val.d;
                out.h2[j] = __float22half2_rn(f2);
            }
            LDST128BITS(b[row * hidden_size + col_idx]) = out.f4;
        }
    }
}

#define binding_splitk_gen(name, pass1_kernel, pass2_kernel)                                                           \
    void name(torch::Tensor a, torch::Tensor b) {                                                                      \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        const int bs = a.size(0);                                                                                      \
        const int lda = a.size(1);                                                                                     \
        const int threads_per_block = 256;                                                                             \
        constexpr int VECS_PER_THREAD = 16;                                                                            \
        int chunk_size = threads_per_block * VECS_PER_THREAD * 8;                                                      \
        int blocks_per_row = (lda + chunk_size - 1) / chunk_size;                                                      \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(a.device());                               \
        auto ws_m = torch::empty({bs, blocks_per_row}, options);                                                       \
        auto ws_d = torch::empty({bs, blocks_per_row}, options);                                                       \
        dim3 grid(blocks_per_row, bs);                                                                                 \
        pass1_kernel<threads_per_block, VECS_PER_THREAD><<<grid, threads_per_block, 0, stream>>>(                      \
            reinterpret_cast<half *>(a.data_ptr()), ws_m.data_ptr<float>(), ws_d.data_ptr<float>(), lda);              \
        pass2_kernel<threads_per_block, VECS_PER_THREAD>                                                               \
            <<<grid, threads_per_block, 0, stream>>>(reinterpret_cast<half *>(a.data_ptr()),                           \
                                                     reinterpret_cast<half *>(b.data_ptr()),                           \
                                                     ws_m.data_ptr<float>(),                                           \
                                                     ws_d.data_ptr<float>(),                                           \
                                                     lda);                                                             \
    }

binding_splitk_gen(softmax_splitk, softmax_grid_pass1, softmax_grid_pass2);
```

Some might ask: why two kernel launches and a temporary workspace allocation? The temporary buffer is my fault — it could be pre-allocated, but I was too lazy to add extra parameters. We're not chasing peak performance here. Demonstration purposes only.

As for the two kernel launches — if you don't use two kernels, you'd need cross-block synchronization. So let me ask: what's the best way to synchronize across multiple blocks? I think launching two kernels on the same stream works just fine. Again, this is for demonstration, not for sweating the details.

Is a single-kernel implementation possible? Yes, actually:

- For example, using `cooperative_groups` with `grid.sync()`, but this requires adjusting the block count since all blocks must reside on SMs simultaneously.
- Alternatively, hand-roll a persistent kernel: launch the maximum number of resident blocks, then use `atomicAdd` on a global variable + a while-loop poll.

Both feel overly complicated and not robust, so let's skip that.

# 4. Benchmark

After all these kernels, let's see how they perform.

## 4.1 Compilation Output

First, let's look at the ptxas info:

```yaml
ptxas info    : 28 bytes gmem
ptxas info    : Compiling entry function '_Z18softmax_grid_pass2ILi256ELi16EEvP6__halfS1_PfS2_i' for 'sm_120'
ptxas info    : Function properties for _Z18softmax_grid_pass2ILi256ELi16EEvP6__halfS1_PfS2_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 40 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 46.738 ms
ptxas info    : Compiling entry function '_Z18softmax_grid_pass1ILi256ELi16EEvP6__halfPfS2_i' for 'sm_120'
ptxas info    : Function properties for _Z18softmax_grid_pass1ILi256ELi16EEvP6__halfPfS2_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 86 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 25.829 ms
ptxas info    : Compiling entry function '_Z24softmax_arbitrary_kernelILi256EEvP6__halfS1_i' for 'sm_120'
ptxas info    : Function properties for _Z24softmax_arbitrary_kernelILi256EEvP6__halfS1_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 36 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 17.098 ms
ptxas info    : Compiling entry function '_Z22softmax_onepass_kernelILi256ELi32ELi24ELi1EEvP6__halfS1_i' for 'sm_120'
ptxas info    : Function properties for _Z22softmax_onepass_kernelILi256ELi32ELi24ELi1EEvP6__halfS1_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 250 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 229.864 ms
ptxas info    : Compiling entry function '_Z22softmax_onepass_kernelILi256ELi8ELi8ELi3EEvP6__halfS1_i' for 'sm_120'
ptxas info    : Function properties for _Z22softmax_onepass_kernelILi256ELi8ELi8ELi3EEvP6__halfS1_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 80 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 30.446 ms
ptxas info    : Compiling entry function '_Z28softmax_fp16x8_packed_kernelILi256EEvP6__halfS1_i' for 'sm_120'
ptxas info    : Function properties for _Z28softmax_fp16x8_packed_kernelILi256EEvP6__halfS1_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 95 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 35.515 ms
ptxas info    : Compiling entry function '_Z21softmax_fp32x4_kernelILi256EEvPfS0_i' for 'sm_120'
ptxas info    : Function properties for _Z21softmax_fp32x4_kernelILi256EEvPfS0_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 55 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 9.114 ms
ptxas info    : Compiling entry function '_Z14softmax_kernelILi256E6__halfEvPT0_S2_i' for 'sm_120'
ptxas info    : Function properties for _Z14softmax_kernelILi256E6__halfEvPT0_S2_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 90 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 14.633 ms
ptxas info    : Compiling entry function '_Z14softmax_kernelILi256EfEvPT0_S1_i' for 'sm_120'
ptxas info    : Function properties for _Z14softmax_kernelILi256EfEvPT0_S1_i
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 86 registers, used 1 barriers, 64 bytes smem
ptxas info    : Compile time = 13.406 ms
```

- The pure register cache kernel uses 95 registers — a bit of a miss. That means only two active blocks (256 / 95 = 2).
  - Some tuning could bring this down to guarantee 3 blocks, but again, too lazy to bother.
- The softmax medium kernel uses 80 registers. I constrained it with `__launch_bounds__` to launch 3 blocks, so the usage is reasonable.
- The softmax extreme kernel truly lives up to its name — 250 registers is maxed out. Of course, heavy unrolling is a major contributor, but we're not sweating the details here.
- Every kernel uses 64 bytes of smem, which is the warp reduce buffer. The smem used as cache is allocated dynamically and thus invisible to the compiler's static analysis.

## 4.2 Performance

### 4.2.1 Small bs + Small hidden_size

```yaml
####################################################################################################
bs: 128, hidden_size: 1024
torch                          mean time: 0.013385 ms
softmax_fp16x8_packed          mean time: 0.011000 ms, speedup: 1.22
softmax_medium                 mean time: 0.010000 ms, speedup: 1.34
softmax_extreme                mean time: 0.033377 ms, speedup: 0.40
softmax_arbitrary              mean time: 0.009563 ms, speedup: 1.40
```

### 4.2.2 Large bs, Small hidden_size

```yaml
####################################################################################################
bs: 2048, hidden_size: 1024
torch                          mean time: 0.017927 ms
softmax_fp16x8_packed          mean time: 0.070659 ms, speedup: 0.25
softmax_medium                 mean time: 0.051850 ms, speedup: 0.35
softmax_extreme                mean time: 0.430039 ms, speedup: 0.04
softmax_arbitrary              mean time: 0.019672 ms, speedup: 0.91
####################################################################################################
bs: 2048, hidden_size: 2048
torch                          mean time: 0.046208 ms
softmax_fp16x8_packed          mean time: 0.075269 ms, speedup: 0.61
softmax_medium                 mean time: 0.056936 ms, speedup: 0.81
softmax_extreme                mean time: 0.440854 ms, speedup: 0.10
softmax_arbitrary              mean time: 0.021029 ms, speedup: 2.20
####################################################################################################
bs: 2048, hidden_size: 4096
torch                          mean time: 0.154837 ms
softmax_fp16x8_packed          mean time: 0.086135 ms, speedup: 1.80
softmax_medium                 mean time: 0.065064 ms, speedup: 2.38
softmax_extreme                mean time: 0.457006 ms, speedup: 0.34
softmax_arbitrary              mean time: 0.036486 ms, speedup: 4.24
####################################################################################################
bs: 2048, hidden_size: 8192
torch                          mean time: 0.324745 ms
softmax_fp16x8_packed          mean time: 0.212331 ms, speedup: 1.53
softmax_medium                 mean time: 0.212712 ms, speedup: 1.53
softmax_extreme                mean time: 0.488978 ms, speedup: 0.66
softmax_arbitrary              mean time: 0.208171 ms, speedup: 1.56
```

### 4.2.3 Small bs, Large hidden_size

```yaml
####################################################################################################
bs: 4, hidden_size: 16384
torch                          mean time: 0.012369 ms
softmax_medium                 mean time: 0.011401 ms, speedup: 1.08
softmax_extreme                mean time: 0.013799 ms, speedup: 0.90
softmax_arbitrary              mean time: 0.010522 ms, speedup: 1.18
softmax_splitk                 mean time: 0.016911 ms, speedup: 0.73
####################################################################################################
bs: 4, hidden_size: 32768
torch                          mean time: 0.033266 ms
softmax_medium                 mean time: 0.015141 ms, speedup: 2.20
softmax_extreme                mean time: 0.014320 ms, speedup: 2.32
softmax_arbitrary              mean time: 0.011193 ms, speedup: 2.97
softmax_splitk                 mean time: 0.020862 ms, speedup: 1.59
####################################################################################################
bs: 4, hidden_size: 65536
torch                          mean time: 0.016672 ms
softmax_extreme                mean time: 0.014532 ms, speedup: 1.15
softmax_arbitrary              mean time: 0.016860 ms, speedup: 0.99
softmax_splitk                 mean time: 0.046560 ms, speedup: 0.36
####################################################################################################
bs: 4, hidden_size: 114688
torch                          mean time: 0.023799 ms
softmax_extreme                mean time: 0.040429 ms, speedup: 0.59
softmax_arbitrary              mean time: 0.023173 ms, speedup: 1.03
softmax_splitk                 mean time: 0.020216 ms, speedup: 1.18
####################################################################################################
bs: 4, hidden_size: 262144
torch                          mean time: 0.043840 ms
softmax_arbitrary              mean time: 0.043765 ms, speedup: 1.00
softmax_splitk                 mean time: 0.045702 ms, speedup: 0.96
####################################################################################################
bs: 4, hidden_size: 1048576
torch                          mean time: 0.193523 ms
softmax_arbitrary              mean time: 0.158823 ms, speedup: 1.22
softmax_splitk                 mean time: 0.026112 ms, speedup: 7.41
####################################################################################################
bs: 4, hidden_size: 8388608
torch                          mean time: 2.036284 ms
softmax_arbitrary              mean time: 2.943902 ms, speedup: 0.69
softmax_splitk                 mean time: 0.689904 ms, speedup: 2.95
####################################################################################################
bs: 4, hidden_size: 33554432
torch                          mean time: 7.662010 ms
softmax_arbitrary              mean time: 10.514376 ms, speedup: 0.73
softmax_splitk                 mean time: 2.372789 ms, speedup: 3.23
```

## 4.3 Summary

Feeling a bit confused after seeing the benchmark results? Here's a summary:

- The two-pass Vanilla kernel is faster than the carefully designed alternatives in most small-scale scenarios.
  - When hidden_size is small, the data read in Pass 1 stays resident in L2 cache. The subsequent Pass 2 read is effectively an L2 access. In contrast, the one-pass approach forces data into registers, tanking occupancy and defeating the purpose.
  - This is the classic **L2 backstop effect** for memory-bound operators on modern architectures.
- Excessive unrolling causes register usage to explode, drops occupancy, and hurts overall performance.
- For very large hidden_size, split-K delivers significant speedups.

So here comes the interviewer's question: When do you use which kernel? What factors influence the choice? How do you optimize? If you had to write a heuristic dispatch algorithm, how would you do it? That's a discussion I'll leave to the reader~ haha~

- Key hints: Are there code-level improvements? Unroll tuning (manual unrolling), register vs. occupancy trade-offs, whether to intentionally spill registers, L2 capacity tipping point estimation, etc.

# 5. Conclusion

To reiterate, none of the kernels presented here are fully tuned for peak performance — they are for illustration purposes only (so suboptimal or even poor performance is entirely possible). No offense intended, and thank you~

The main takeaway I'd like to share is the approach of manually caching in registers and shared memory, the L2-backstopped kernel implementation that supports arbitrary lengths, and finally the split-K optimization. I hope you find it useful.

All kernels and test code are available on GitHub:

That's all.
