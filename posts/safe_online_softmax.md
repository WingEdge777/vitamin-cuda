# [CUDA 优化实战] safe online softmax - 面试必问：任意 hidden_size、one/two pass、trade-off、split-k

# 0. 序 - 背景

softmax 是深度学习中常用算子，在几乎所有机器学习领域常用来做置信度/权重/概率输出预测。可以说没有 softmax，模型就没有输出（embedding 除外）。

> 本人对 softmax 也印象深刻，因为当年校招入职时，主管面最后就让我用 numpy 白板写了个 softmax forward

而在当今大型分类模型或 LLM 中，softmax 已经开始接受各种角度的蹂躏。几乎堪比 gemm 一样，要考虑在各种 case 下进行调优。

如果我是面试官，我必问候选人对各个场景 softmax 的调优，因此本文是一个 softmax 各种实现的方法和考量集合。

> 从没有最佳 kernel，只有最合适的 kernel
> ---------------------------------- altum sonatur（随便加点拉丁语，就会显得高大上）

！注意，本文所列出的所有 kernel 只是为了起到演示效果，并没有经过极致的手工优化。默认数据全部已经对齐到 128B。

本文整体会给出两类 kernel 实现，kernel 大纲如下：

- one-pass
  - softmax fp16x8 版 (fp16 向量化，pure register cache，packed r/w)
  - softmax medium fp16 版 (fp16 向量化，medium register+smem cache usage, packed r/w)
  - softmax extreme fp16 版 (fp16 向量化，maximum register+smem cache usage, packed r/w)
- two-pass
  - softmax arbitrary fp16 版 (fp16 向量化，packed r/w)
  - softmax split-k fp16 版 (fp16 向量化，packed r/w)

这里的 pass 是指对 input 遍历的次数。

- one-pass kernel： 主要思想是利用 register 和 smem 做 input 的 cache，当 reduce 完 max 和 exp_sum 后，计算 output 时 input 直接从 register 和 smem 取；
- two-pass kernel：第一就是朴素实现，不 cache 输入，读多少算多少；第二就是如今流行的 split-k 思想，在 bs 较小而 dim 特别大时，对 dim 进行分块，块内 block reduce 一次，然后对所有 block 的结果再 reduce 一次；

当然，以上实现都是基于 safe 的前提，online 或者不 online，一看规模，二看 kernel 的具体环节。

# 1. safe online softmax

这年头大家应该都听过 safe online softmax (flash attn 基石），所以这里就简单给三个 numpy 版的代码对比+注释说明：
softmax：

```python
import numpy as np

def softmax(x):
    """
    基础版本，容易发生数值溢出
    公式：exp(x) / sum(exp(x))
    """
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

safe softmax：

```python
def safe_softmax(x):
    """
    安全版本，利用平移不变性防止溢出
    公式：exp(x - max) / sum(exp(x - max))
    """
    # Pass 1: 找全局最大值
    x_max = np.max(x, axis=-1, keepdims=True)

    # Pass 2: 减去最大值后计算 exp 并归一化
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

safe online softmax:

```python
def safe_online_softmax(x, block_size=256):
    """
    Online 版本
    """
    batch_size, hidden_size = x.shape

    # 存放全局的 max (m) 和 sum (d)
    m = np.full((batch_size, 1), -np.inf)
    d = np.zeros((batch_size, 1))

    # Pass 1: 分块 Online 计算全局 Max 和 Sum
    for i in range(0, hidden_size, block_size):
        chunk = x[:, i:i+block_size]

        # 找当前块的局部最大值
        m_local = np.max(chunk, axis=-1, keepdims=True)

        # 核心 Online 更新逻辑
        m_new = np.maximum(m, m_local)

        # 计算当前块以 m_new 为基准的局部 sum
        d_local = np.sum(np.exp(chunk - m_new), axis=-1, keepdims=True)

        # 缩放历史 sum (d)，并加上当前的局部 sum
        d = d * np.exp(m - m_new) + d_local

        # 更新全局 max
        m = m_new

    # Pass 2: 利用算好的全局 m 和 d 写回结果
    out = np.zeros_like(x)
    for i in range(0, hidden_size, block_size):
        chunk = x[:, i:i+block_size]
        out[:, i:i+block_size] = np.exp(chunk - m) / d

    return out
```

# 2. one-pass kernels

## 2.1 pure register cache

这里假设输入是二维的`[batch_size, hidden_size]`，朴素想法就是一个 block 处理一个 batch。

首先，block size 我直接选了 256，别问为什么，我这个卡就适合 128 和 256，我喜欢 256.

然后根据我的寄存器总量（65536），算一下每个线程需要最多能用多少寄存器(256)，考虑到 Occupancy 别太低（别让 sm 就一个 block 甚至一个都没有，然后光等着他干活，当然极端情况下也是可以的，这也叫 persistent kernel），所以我选了 64 个寄存器

这个选择并不是很好，因为考虑到 kernel 肯定还有指针，偏移量，临时计算 buffer 等等寄存器用量，估计会接近 80~100，所以活跃 block 只能 2~3 个，但考虑到只是演示用途，并不追求极致性能。而且 64 个 32 位寄存器做 cache， 我就可以覆盖 hidden_size<=8192 内的所有 case，所以就定了 64。

直接上代码：

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

这里在读取 input 的时候，我们直接循环求了单线程内的最大值。然后在 warp reduce 阶段才用到 online softmax 的思想（反复缩放）。

### 2.1.1 warp reduce

warp reduce 我们用到了 `__shfl_xor_sync`  warp 级原语，用于 warp 内直接进行同步的寄存器变量交换，比如：

```cpp
for (int mask = warp_size >> 1; mask > 0; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
}
```

`__shfl_xor_sync`的逻辑是线程 id 可以取到线程 id `id^mask` 的 val。
因此，这个循环会执行五次：

- 第一次 0~15 和 16~31 线程会一一对应的交换值并取最大值
- 第二次 0~7 和 8~15，16~23 和 24~31
- 以此类推...

这个过程又叫经典的蝶形变换，感兴趣的可以看`__shfl_xor_sync`官方文档：<https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=__shfl_xor_sync#warp-shuffle-functions>

或自行搜索其他博客学习细节

## 2.2 register + smem cache

### 2.2.1 medium kernel

这个 kernel 目标是同时用上 register 和 smem，根据我的显卡限制，单个 block smem 最大使用量 48KB，因此最多缓存 48*1024 / 2 / 256 = 96 个 half。
但是 block reduce 还要使用一点点 smem，所以我不能开满，同时我们这里是 medium 版，也是演示用途不是为了测试什么极限场景，因此我选了和上面寄存器用量一致的大小 64 个 half，
这样一个 block 占用 64*256*2 = 32kB，我的 sm 总 smem 量是 100kB，所以可以有 3 个活跃的 block 也和上面寄存器的硬件限制接近。

这个配置下，最大支持 hidden_size 到 32768

ok，分析完硬件，本来应该给代码的，但是由于这里我为了减少代码量，把 medium 版和 extreme 版合为一，通过模板参数区分了，所以下一小节见~

### 2.2.2 extreme kernel

所谓 extreme，就是为了在 one-pass 的情况下 handle 尽量大的 hidden_size，因此我们要把寄存器和 smem 用量拉爆。

- 先看 smem：这个比较好计算，一共 100KB，减掉一点点 warp reduce 的用量，我通过验算凑个整取了 96KB，因此可以缓存 96*1024 / 256 / 2 = 192 个 half（24 个 float4）
  - 注意，这里由于要打破单个 block 48K 的上限，所以只能使用动态共享内存，并通过 cuda 魔法 `cudaFuncSetAttribute` 设置共享内存大小。
- 再看寄存器：这里要说明一下，虽然前面算了单线程上限是 256 个寄存器，但是我们的代码里用了大量的 unroll，同时循环内有多次类型转换、max/exp 计算等，寄存器开销实在过大。经过验算我最终定了 128 个寄存器，也就是 256 个 half（32 个 float4）
  - 注：实测使用了 250 个寄存器

这个配置下，最大支持 hidden_size 到 114688

ok，现在可以上代码了：

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

代码看起来有点长，不过其实是对 register 和 smem 做相同的处理，相关逻辑也是可以抽出函数来的，只是我懒得写了，所以就先这样了~见谅~

# 3. two-pass

## 3.1 Vanilla 实现支持任意 hidden_size

既然 two-pass，那肯定是要支持任意大小的（别爆显存），这里给一个朴素实现，就是向量化加载数据，然后用 online 的方式更新最大值和 sum

代码如下：

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

可以看到，代码其实很干净，就是向量化读取数据进来，然后用 online 的方式更新单线内的最大值。再做一次 block reduce。最后写回 b 时，再读一遍 a。

看起来很朴素，实际上也确实很朴素。但实测性能其实挺不错，因为在 hidden_size 不大的情况下，读一遍 a，a 就被缓存到 L2 了，第二遍相当于是从 L2 读取。

## 3.2 split-k kernel

split-k 是为了加速超大 hidden_size（但 batch size 较小）情况下的计算速度，核心思想就是沿着 hidden_size 方向切块，一个 block 计算一个 chunk 内的局部最大值和 sum，写回 global memory。

然后再启动一次 kernel，读取上次的整合所有 chunk 的结果，再做一次 reduce，读一遍输入，计算输出。

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

有人说，你这怎么启动两个 kernel，还申请临时空间。临时空间的 buffer 是我的锅，这个其实是可以提前申请的，但我懒得再传参数了，所以没写。咱也不是追求极限性能。起到演示效果即可。

至于两次 kernel launch 嘛，如果不用两个 kernel，那么就涉及到多个 block 之间的同步了。试问一下，多个 block 之间最佳的同步方法是什么？我觉得分两次 kernel 放到一条 stream 上执行就挺好的。还是那句话，咱是为了演示效果，不扣细节。

有没有单 kernel 实现？其实也是有的

- 比如用`cooperative_groups`, 然后用 grid.sync()，但这个要配合 block 数量调整，因为要强制所以 block 驻留 SM。
- 再就是自己手搓 persistent kernel，启动最大的 block 驻留数量的线程，然后 atomiAdd 全局变量 + while 轮询。

我感觉都太麻烦了，而且不 robust，还是算了吧。

# 4. benchmark

说了这么多种 kernel，看看效果呗

## 4.1 编译输出

先看下 ptxas info：

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

- 纯寄存器缓存输入 kernel 的寄存器数量用到了 95 个，有点失败，那么就只能有两个 block（256/95 = 2）活跃了
  - 其实调一调还是可以压低些保障 3 个 block 的，还是懒得弄了；
- softmax medium 寄存器 80 个，这个我用`__launch_bound__`限制住了发射 3 个 block，所以用量还行；
- softmax extreme 确实 extreme 了，250 个寄存器用量已经拉爆，当然也有 unroll 太多占用大量寄存器用量的原因，不过这里不扣细节了；
- 每个 kernel 都用了 64 bytes smem，这个其实是 block reduce 中 smem 的用量，用作 cache 的 smem 由于是使用的动态共享数组，编译器无法统计到；

## 4.2 性能

### 4.2.1 small bs + small hidden_size

```yaml
####################################################################################################
bs: 128, hidden_size: 1024
torch                          mean time: 0.013385 ms
softmax_fp16x8_packed          mean time: 0.011000 ms, speedup: 1.22
softmax_medium                 mean time: 0.010000 ms, speedup: 1.34
softmax_extreme                mean time: 0.033377 ms, speedup: 0.40
softmax_arbitrary              mean time: 0.009563 ms, speedup: 1.40
```

### 4.2.2 large bs，small hidden_size

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

### 4.2.3 small bs，large hidden_size

```yaml
####################################################################################################
bs: 4, hidden_size: 16384
torch                          mean time: 0.010786 ms
softmax_medium                 mean time: 0.009192 ms, speedup: 1.17
softmax_extreme                mean time: 0.011900 ms, speedup: 0.91
softmax_arbitrary              mean time: 0.009755 ms, speedup: 1.11
softmax_splitk                 mean time: 0.016945 ms, speedup: 0.64
####################################################################################################
bs: 4, hidden_size: 32768
torch                          mean time: 0.027404 ms
softmax_medium                 mean time: 0.009107 ms, speedup: 3.01
softmax_extreme                mean time: 0.014393 ms, speedup: 1.90
softmax_arbitrary              mean time: 0.013321 ms, speedup: 2.06
softmax_splitk                 mean time: 0.020143 ms, speedup: 1.36
####################################################################################################
bs: 4, hidden_size: 65536
torch                          mean time: 0.014817 ms
softmax_extreme                mean time: 0.012640 ms, speedup: 1.17
softmax_arbitrary              mean time: 0.015067 ms, speedup: 0.98
softmax_splitk                 mean time: 0.018317 ms, speedup: 0.81
####################################################################################################
bs: 4, hidden_size: 114688
torch                          mean time: 0.022233 ms
softmax_extreme                mean time: 0.015460 ms, speedup: 1.44
softmax_arbitrary              mean time: 0.021300 ms, speedup: 1.04
softmax_splitk                 mean time: 0.021642 ms, speedup: 1.03
####################################################################################################
bs: 4, hidden_size: 262144
torch                          mean time: 0.044464 ms
softmax_arbitrary              mean time: 0.041788 ms, speedup: 1.06
softmax_splitk                 mean time: 0.024179 ms, speedup: 1.84
####################################################################################################
bs: 4, hidden_size: 1048576
torch                          mean time: 0.160862 ms
softmax_arbitrary              mean time: 0.157954 ms, speedup: 1.02
softmax_splitk                 mean time: 0.026998 ms, speedup: 5.96
####################################################################################################
bs: 4, hidden_size: 8388608
torch                          mean time: 1.740836 ms
softmax_arbitrary              mean time: 2.546787 ms, speedup: 0.68
softmax_splitk                 mean time: 0.605970 ms, speedup: 2.87
####################################################################################################
bs: 4, hidden_size: 33554432
torch                          mean time: 6.858849 ms
softmax_arbitrary              mean time: 10.122205 ms, speedup: 0.68
softmax_splitk                 mean time: 2.395403 ms, speedup: 2.86
```

## 4.3 总结

看完 benchmark 结果是不是有点懵逼？总结一下：

- two-pass 的 Vanilla 在大多数小场景下比精心设计的其他 kernel 还快；
  - 在 Hidden Size 较小时，Pass 1 读取的数据驻留在 L2 Cache 中。随后 Pass 2的读取等价于L2访问。相比之下，One-Pass 为了强行把数据塞进寄存器，导致 Occupancy 暴跌，得不偿失。
  - 这就是典型的 Memory-Bound 算子在现代架构下的 L2 兜底效应。
- 过度的 unroll 导致寄存器用量爆炸，Occupancy 下降，影响整体性能；
- 超长 hidden_size splitk 确实效果显著；

所以啊，这时面试官的问题就来了：什么情况下用什么 kernel？受什么因素影响？如何优化？如果让你写一个启发式的 dispatch 算法该怎么做？这个，就留给大家讨论吧~哈哈~

- 关键点提示：纯代码层面有没有改善点，unroll限制调优(手动展开)，寄存器和Occupancy的trade-off，要不要主动寄存器溢出，L2容量临界点判断等等

# 5. 结束

重申，所有 kernel 都不是经过调优的最佳 kernel，只做演示意图（所以性能不合预期，甚至很拉跨都是有可能的）。不喜勿喷，感谢~

主要还是想向大家分享一下，手动以寄存器、smem 做 cache 的思路，和支持任意长度的 L2 兜底的 kernel 实现，以及最后 split-k 的优化实现，希望大家有所收获；

所有 kernel 和测试代码可以从 github 获取：

以上
