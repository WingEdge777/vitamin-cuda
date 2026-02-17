# [CUDA 优化实战] 矩阵转置-从 Padding 到 XOR Swizzle：CUDA 共享内存优化的艺术

矩阵转置（Transpose）是深度学习和高性能计算中极其基础的操作。看似简单的坐标交换 $B[y][x] = A[x][y]$，在 CPU 上可能只是两层循环，但在 GPU 这种吞吐导向的架构上，访存模式（Memory Access Pattern） 往往比计算逻辑更能决定性能的生死。如何写出一个高效的 Transpose Kernel 往往是考察 CUDA 工程师功底的试金石。

本文将基于一份实际的 CUDA 代码，带大家一步步从最朴素的实现，演进到结合了 Shared Memory、Bank Conflict Free (Padding) 以及 XOR Swizzle 机制的高效版本，并在这个过程中解释共享内存的 bank conflict 的由来以及简单的 swizzle 机制避免 Bank Conflict 的原理

## 0. 为什么转置很难？访存的合并与非合并

在 GPU 全局内存（Global Memory）中，最佳的访存模式是 Coalesced Access（合并访问）。简单来说，当一个 Warp 中的 32 个线程访问连续的内存地址时，GPU 可以将这些访问合并成极少量的内存事务。

矩阵转置的天然矛盾在于：

- 如果我们按行读取（合并读取），转置后写入时必然是按列写入（非合并写入，Strided Access）
- 如果我们按列读取（非合并读取），转置后写入时虽是按行写入（合并写入），但读取端又慢了

## 1. 朴素实现

我们在代码中实现了两个基准版本，分别为合并读取（跨步读取）和合并写入

1. transpose_coalesced_read_kernel: 合并读，离散写。
2. transpose_coalesced_write_kernel: 离散读，合并写

```cpp

const int naive_tiling_size = 16;

// naive coalesced read transpose
__global__ void transpose_coalesced_read_kernel(float *a, float *b, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        b[x * height + y] = a[y * width + x];
    }
}
// launch 代码
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

// naive coalesced write transpose
__global__ void transpose_coalesced_write_kernel(float *a, float *b, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < height && y < width) {
        b[y * height + x] = a[x * width + y];
    }
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
```

**性能瓶颈**：无论哪种方式，总有一端会因为跨步访问（Stride Access）导致显存带宽利用率低下。对于大型矩阵，这简直是带宽杀手。

本人这里的 naive 实现，blockDim 设置为 16x16。 也许会有人问，为什么不用 32x32?
因为 16x16=256, 32x32=1024, 而我的显卡 5060 每个 SM 最多驻留 1536 个线程（这里就体现了我第一篇文章的内容的重要性了，你首先要了解你的硬件）。如果设置为 32x32, 那么一个 SM 只能放下一个 block, Occupancy 只有 2/3, 这会严重影响流水线效率；而设置为 16x16, 一个 SM 可以同时有 6 个 block 共 1536 个线程活跃，Occupancy 100%.

此外还有共享内存的占用，当然32x32的分块也只占用4k，我的显卡 SM 片上内存有100KB，4K 的占用，可以允许充足的block数量，这里无需多加担心。

好奇的人又会问了，那能不能设置为 8x8, 既可以避免边缘 case 冗余的线程数量，而且活跃 block 数量更多了，又能更好的隐藏时延。
答案是，可以设置为 8x8, 但是要知道， tiling_size x tiling_size 的访问模式，本身在一个 warp（32 线程）内已经是两次分段访问了，这种半 warp 的访问模式，通常是可以接受的，但如果再进一步细化 block, 会使得访问的连续性更差，影响 cache；其次 block 数量变多，发射访问显存的指令数也变多，SM 的调度开销也变大。当然，感兴趣的朋友可以自己尝试一下，用 8x8 的 block size, 看看性能如何。

## 2. 引入 Shared Memery：（tiling cache）

为了解决 Global Memory 的非合并访问问题，标准的解法是利用 Shared Memory (Smem) 作为 cache 中转。Smem 是片上高速缓存，带宽极高，且对随机访问的容忍度远高于 Global Memory。
核心思想（Tiling）：

- 将 input 大矩阵切分成 $32 \times 32$ 的小块（Tile）
- 线程块将数据从 GMEM 合并读取 到 Smem
- 坐标变换从 Smem 读取出列数据
- 将列数据按行合并写入到 output

kernel 代码如下：

```cpp
const int tiling_size = 32;
const int tiling_row = 8;

// transpose with Smem
__global__ void transpose_smem_kernel(float *a, float *b, int width, int height) {
    __shared__ float tile[tiling_size][tiling_size];
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int x = bx * tiling_size + tx;
    int y = by * tiling_size + ty;

    // 避免每次读取/写入都要判断是否越界，提前做好边界判断
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

// launch kernel
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
            a.data_ptr<float>(), b.data_ptr<float>(), width, height);          \
    }
```

引入的这个 kernel 看起来复杂了许多，让我们一步一步拆解下实现细节；
首先，我们定义了 `tiling_size` 为 32，`tiling_row` 为 8，这表示我们将使用一个 32x8 的线程块来处理数据块。数据分块依然是以 32x32 的 tile 为单位进行处理，但线程 block 改为了 32x8，这样既保证了 warp 线程访问的连续性，又将 block 线程数量定在了 256，占用率不会变化，但同时由于线程行减小，同时 grid 设置不变，因此每一个 block 要处理 4 行数据，这就是代码中 for 循环的`j += tiling_row`的由来。

其次，我们使用了共享内存来存储 tile 数据，这样可以减少全局内存的访问次数，提高性能。在处理数据时，我们首先将数据从全局内存加载到共享内存中，
`tile[ty + j][tx] = a[(y + j) * width + x];` ty 是 block id，tx 是 thread id，y 是 block 在 height 方向的偏移，x 是 block 在 width 方向的偏移，这样一个 warp 中，y 是固定的，x 是连续的行，正好可以将 tile 的一行数据写入到 Smem。

写入完共享内存后，我们使用了一个 `__syncthreads()` 来同步所有线程，因为转置读取依赖block内其他warp搬运的数据，必须确保一个 block 内的所有线程数据都写入完成，才能保证读取到正确的数据。

最后，在将数据从共享内存写入全局内存时，我们使用了 `b[(y + j) * height + x] = tile[tx][ty + j];` 写法，这样可以按列从 Smem 读取数据后并在 b 的内存中按行连续存储

### 2.1 Bank Conflict

**问题出现**：前述代码虽然解决了 Global Memory 的合并读写问题，但我们引入了新的问题——Shared Memory Bank Conflict。
首先，解释一下，共享内存是按地址每 4 bytes 划分为 32 个 bank，每个 bank 宽度为 4 字节（32 bits）当多个线程访问同一个 bank 的不同地址时，就会发生 bank conflict，导致性能下降。
在我们的代码中，`tile[ty + j][tx]`的写入模式没问题，但`tile[tx][ty + j]` 的读取模式产生了冲突。32x32 的 tile 读取时，归属 bank id 如下：

```yaml
 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
 ...

```

显然，一个 warp 按列连续读取时，都是同一个 bank，但又不是同一地址，因为会触发 32 次串行的内存请求，性能大幅降低。

### 2.2 解决方案：padding

在共享内存中，可以通过在每行末尾添加 padding 来避免 bank conflict。例如，将 32x32 的数组改为 32x33，这样每地址都会相对上一行偏移一个 bank，正好落在 32 个 bank，bank id 归属如下：

```yaml
 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0
 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1
 2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2
 3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3
 4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4
 5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5
 6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6
 7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7
 8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8
 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9
10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10
11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11
12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12
13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13
14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17
18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18
19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22
23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27
28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28
29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
```

这样就避免了冲突

## 3. 进阶优化：叠加向量化读写 (Float4)

为了进一步榨干带宽，我们可以使用向量化指令 LDS.128 / STS.128，即一次读写 float4。同时原来一个 block 要处理一行数据，这里我们通过线程 id 变换行/列号，让一个线程处理 4 个元素，去除了循环。

```cpp
#define LDST128BITS(x) (*reinterpret_cast<float4*>(&(x)))

// Smem bcf + float4 r/w
__global__ void transpose_smem_packed_bcf_kernel(float *a, float *b, int width, int height) {
    __shared__ float tile[tiling_size][tiling_size + 1];
    int tid = threadIdx.x + threadIdx.y * blockDim.x; // [0, 256]

    int sx = tid % 8; // 这里实际上是 column index / 4
    int sy = tid / 8; // 这里是 row index

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
```

**思考**: 为什么 Shared Memory 不使用向量化写入？
细心的读者可能发现，我们将数据从 Global Memory 向量化读取 (float4) 后，拆解成了 4 个标量写入 Shared Memory。

既然 Shared Memory 也支持向量化访问（LDS.128/STS.128），为什么不直接 float4 写入？
这里面临两个棘手的冲突：

- Padding 对齐问题：为了消除 Bank Conflict 引入的 Padding (32x33) 破坏了 128-bit (16 bytes) 的地址对齐要求，导致无法直接使用向量化指令
  - 如：第二行首地址 (1*33 + 0)*4 = 132 % 16 = 4 != 0, 不满足地址对齐要求
- Swizzle 复杂性：如果去除 Padding 依靠 Swizzle，标准的 XOR Swizzle 在连续写入 float4 时，四个分量可能会被映射到非连续的地址，甚至导致写入时的 Bank Conflict（虽然不如读取时严重）
  - 注：可以设计再稍微复杂的 layout + Swizzle 模式，实质上就是对将数据 pack 后的索引下标进行 swizzle，这样就可以实现 Shared Memory 无冲突的向量化读写，但这超出了本文的基础范畴，不做详细描写

## 4. 终极方案：向量化读写 (Float4) + Smem swizzling 读写

如果说 Padding 是通过‘扩建停车场’（牺牲空间）来避免拥堵，那么 Swizzle 就是一位天才的‘交通指挥员’。它不占用额外空间，而是通过一套数学映射规则（XOR），让原本可能在物理上扎堆的车辆，在逻辑上‘错峰’停放，既保留了标准的车位间距（保证对齐），又消除了冲突。

在现代高性能库（如 CuDNN, CUTLASS）中，Swizzling 是处理 Shared Memory 访问冲突的高级技巧。与其浪费空间做 Padding，不如通过数学变换（异或操作 XOR）打乱数据的存储布局。而且在现代 GPU 架构中，异步拷贝指令（cp.async）和 Tensor Core 的某些布局天然就更适配这种打乱的模式。

这里做演示用途，对 global memory 做向量化读写，对 Smem 做 swizzling 读写。这样既保证了对齐，又避免了 bank conflict（实际上可以实现 Smem 的 swizzling 向量化读写，比如 cutlass 就提供了相关的实现）

```cpp
// Smem swizzle bcf + float4 r/w
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
```

### 4.1 swizzle 机制

swizzle 本质上是通过数学变换，将数据在 shared memory 中的物理存储位置进行重新排列，以避免 bank conflict。在读取时，通过相同的变换来恢复数据的原始顺序。这种技术在需要高性能计算中很常见。代码中使用的是 xor swizzle 地址变换，其利用了 xor 在二进制域上的双射性。

逻辑地址到物理地址的变换方式为 (x, y) --> (x, y ^ x)

我们可以简单写一个脚本验证一下地址变换后的情况

```python
for i in range(32):
    for j in range(32):
        print(f"{i^j:2d} ", end="")
    print()

```

输出：

```yaml
 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
 1  0  3  2  5  4  7  6  9  8 11 10 13 12 15 14 17 16 19 18 21 20 23 22 25 24 27 26 29 28 31 30
 2  3  0  1  6  7  4  5 10 11  8  9 14 15 12 13 18 19 16 17 22 23 20 21 26 27 24 25 30 31 28 29
 3  2  1  0  7  6  5  4 11 10  9  8 15 14 13 12 19 18 17 16 23 22 21 20 27 26 25 24 31 30 29 28
 4  5  6  7  0  1  2  3 12 13 14 15  8  9 10 11 20 21 22 23 16 17 18 19 28 29 30 31 24 25 26 27
 5  4  7  6  1  0  3  2 13 12 15 14  9  8 11 10 21 20 23 22 17 16 19 18 29 28 31 30 25 24 27 26
 6  7  4  5  2  3  0  1 14 15 12 13 10 11  8  9 22 23 20 21 18 19 16 17 30 31 28 29 26 27 24 25
 7  6  5  4  3  2  1  0 15 14 13 12 11 10  9  8 23 22 21 20 19 18 17 16 31 30 29 28 27 26 25 24
 8  9 10 11 12 13 14 15  0  1  2  3  4  5  6  7 24 25 26 27 28 29 30 31 16 17 18 19 20 21 22 23
 9  8 11 10 13 12 15 14  1  0  3  2  5  4  7  6 25 24 27 26 29 28 31 30 17 16 19 18 21 20 23 22
10 11  8  9 14 15 12 13  2  3  0  1  6  7  4  5 26 27 24 25 30 31 28 29 18 19 16 17 22 23 20 21
11 10  9  8 15 14 13 12  3  2  1  0  7  6  5  4 27 26 25 24 31 30 29 28 19 18 17 16 23 22 21 20
12 13 14 15  8  9 10 11  4  5  6  7  0  1  2  3 28 29 30 31 24 25 26 27 20 21 22 23 16 17 18 19
13 12 15 14  9  8 11 10  5  4  7  6  1  0  3  2 29 28 31 30 25 24 27 26 21 20 23 22 17 16 19 18
14 15 12 13 10 11  8  9  6  7  4  5  2  3  0  1 30 31 28 29 26 27 24 25 22 23 20 21 18 19 16 17
15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0 31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16
16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
17 16 19 18 21 20 23 22 25 24 27 26 29 28 31 30  1  0  3  2  5  4  7  6  9  8 11 10 13 12 15 14
18 19 16 17 22 23 20 21 26 27 24 25 30 31 28 29  2  3  0  1  6  7  4  5 10 11  8  9 14 15 12 13
19 18 17 16 23 22 21 20 27 26 25 24 31 30 29 28  3  2  1  0  7  6  5  4 11 10  9  8 15 14 13 12
20 21 22 23 16 17 18 19 28 29 30 31 24 25 26 27  4  5  6  7  0  1  2  3 12 13 14 15  8  9 10 11
21 20 23 22 17 16 19 18 29 28 31 30 25 24 27 26  5  4  7  6  1  0  3  2 13 12 15 14  9  8 11 10
22 23 20 21 18 19 16 17 30 31 28 29 26 27 24 25  6  7  4  5  2  3  0  1 14 15 12 13 10 11  8  9
23 22 21 20 19 18 17 16 31 30 29 28 27 26 25 24  7  6  5  4  3  2  1  0 15 14 13 12 11 10  9  8
24 25 26 27 28 29 30 31 16 17 18 19 20 21 22 23  8  9 10 11 12 13 14 15  0  1  2  3  4  5  6  7
25 24 27 26 29 28 31 30 17 16 19 18 21 20 23 22  9  8 11 10 13 12 15 14  1  0  3  2  5  4  7  6
26 27 24 25 30 31 28 29 18 19 16 17 22 23 20 21 10 11  8  9 14 15 12 13  2  3  0  1  6  7  4  5
27 26 25 24 31 30 29 28 19 18 17 16 23 22 21 20 11 10  9  8 15 14 13 12  3  2  1  0  7  6  5  4
28 29 30 31 24 25 26 27 20 21 22 23 16 17 18 19 12 13 14 15  8  9 10 11  4  5  6  7  0  1  2  3
29 28 31 30 25 24 27 26 21 20 23 22 17 16 19 18 13 12 15 14  9  8 11 10  5  4  7  6  1  0  3  2
30 31 28 29 26 27 24 25 22 23 20 21 18 19 16 17 14 15 12 13 10 11  8  9  6  7  4  5  2  3  0  1
31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0
```

可以看到，对于任意行列，都是 0~31，因此变换后的地址都属于不同 bank，不会有 bank conflict 问题

#### 4.1.1 xor 数学原理

你可能会问：“按列读取的时候，为什么 Row 变的时候，(Col ^ Row) 一定会生成 0-31 之间不重复的数字？”

简单来说，XOR 运算像是一个不进位的加法，它保证了只要输入不同，在这个特定的变换下输出也一定不同，就像洗牌一样打乱了顺序但没有弄丢任何一张牌。

从数学上来说，这是由 XOR 的代数性质决定的：XOR 是由“可逆”的变换组成的群（Group）。

对于任何固定的常数 $C$（在这里 $C$ 就是列号 Col），函数 $f(x) = x \oplus C$ 是一个 双射（Bijection），也就是一一对应的映射。

- 证明：
  - 假设有两个不同的行号 $x$ 和 $y$ ($x \neq y$)。
  - 如果它们映射到了同一个 Bank，即 $x \oplus C = y \oplus C$。
  - 两边同时异或 $C$
$$(x \oplus C) \oplus C = (y \oplus C) \oplus C$$

$$x \oplus (C \oplus C) = y \oplus (C \oplus C)$$
$$x \oplus 0 = y \oplus 0$$
$$x = y$$

- 这与假设 $x \neq y$ 矛盾。

**结论**： 只要输入的行号 $Row$ 是不同的（0-31），输出的 Bank ID $\text{Row} \oplus \text{Col}$ 就一定是不同的。它只是把 0-31 这组数字的顺序打乱了（Permutation），但绝对不会让两个不同的输入挤到同一个输出里。

#### 4.1.2 更直观的 xor 位运算理解

例如：

- Col = x 时（固定）
- 当 Row 从 000... 便利到 111...，bank_id = Row^Col。

想象一下，异或实质上是一个无进位的加法运算，那么 Row 的任意一位 bit 变动，都会导致异或结果对应 bit 变动

因此 bank_id 会遍历 0 到 31，不会重复，不会冲突

#### 4.1.3 使用其他 swizzling 方式

- 除了 xor，还有其他 swizzling 方式，比如：
- shift: `(x, y) --> (x + y) % 32`

还是使用 python 代码打印一下变换后的地址：

```python
for i in range(32):
    for j in range(32):
        print(f"{(i+j)%32:2d} ", end="")
    print()
```

输出

```yaml
 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0
 2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1
 3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2
 4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3
 5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4
 6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5
 7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6
 8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7
 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8
10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9
11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10
12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11
13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12
14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13
15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
17 18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
18 19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17
19 20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18
20 21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
21 22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
22 23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
23 24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22
24 25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
25 26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
26 27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
27 28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
28 29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27
29 30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28
30 31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
31  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
```

细心的朋友可能已经发现，这种变换和 padding 的方式十分类似，只不过一个是主动 padding 移位物理地址，一个是直接取模循环移位了。

取模运算相对位运算更昂贵，所以不常用。

总之，swizzling 相对 padding 方式，节省了 padding 的空间，避免了地址不对齐的问题，是 cutlass 等优化库中常用的一种技巧。

当然 swizzling 也有缺点，就是需要额外的寄存器和计算来实现，对于一些简单场景可能得不偿失。

#### TODO

使用更完美的 swizzling 技巧，做到 Smem 向量化读写（数据 pack）

## 5. 总结

### 5.1 benchmark 测试

最后贴一个和 pytorch 的对比结果，由于本人是在笔记本电脑上做的测试，无法使程序独占显卡做 benchmark，也无法排除桌面应用和操作系统等其他程序的影响，所以挑了典型的结果出来，以展示效果

```yaml
n: 8192, m: 2048
torch                          mean time: 1.132851 ms
transpose_coalesced_read       mean time: 0.519010 ms, speedup: 2.18
transpose_coalesced_write      mean time: 0.506867 ms, speedup: 2.24
transpose_Smem                 mean time: 0.480647 ms, speedup: 2.36
transpose_Smem_bcf             mean time: 0.445889 ms, speedup: 2.54
transpose_Smem_packed_bcf      mean time: 0.418792 ms, speedup: 2.71
transpose_Smem_swizzled_packed mean time: 0.414753 ms, speedup: 2.73
```

可以看到，通过 Swizzle + Vectorization，我们获得了 2.73x 的加速。

加上带宽计算结果:

```yaml
Kernel Name                     Mean Time (ms)    Speedup    Effective Bandwidth
--------------------------------------------------------------------------------
torch.transpose                  1.132851 ms       1.00x      118.5 GB/s
transpose_coalesced_read         0.519010 ms       2.18x      258.6 GB/s
transpose_coalesced_write        0.506867 ms       2.24x      264.8 GB/s
transpose_Smem (Base)            0.480647 ms       2.36x      279.3 GB/s
transpose_Smem_bcf (Padding)     0.445889 ms       2.54x      301.1 GB/s
transpose_Smem_packed_bcf        0.418792 ms       2.71x      320.5 GB/s
transpose_Smem_swizzled_packed   0.414753 ms       2.73x      323.6 GB/s
```

细心的读者可能发现了一个“异常”数据：最终优化版本的有效带宽达到了 323 GB/s。然而，对于我这张卡来说，其 DRAM 物理极限带宽 ~= 192 GB/s

*为什么实测带宽会超过硬件的物理极限？*

- **数据量分析**：本次测试的矩阵大小为 $8192 \times 2048 \times 4 \text{ bytes} = 64 \text{ MB}$。
- **缓存机制**：现代 GPU 拥有巨大的 L2 Cache（我这张卡有32MB）。
- **结论**：虽然总数据量可能超过 L2 大小，但高效的 Tiling 和 Swizzle 极大提升了 Cache Line 的利用率和局部性，使得有效带宽逼近甚至在短时突发中超过了 DRAM 的理论平均值。

因此，这里的 323.6 GB/s 实际上反映的是 L2 Cache + DRAM 的混合吞吐能力。这也侧面印证了我们的优化策略（Tiling + Coalesced Access）极大地提升了数据的局部性（Data Locality），使得 Cache 命中率大幅提高。

### 5.2 优化手段的总结

从本文，我们学到了 CUDA 优化的核心路径：

- Global Memory: 必须保证 Coalesced Access（合并访问）。
- Vectorization: 使用 float4 等类型提升带宽吞吐
- Shared Memory: 用作数据中转站（Corner Turn），将非合并访问转化为合并访问。
  - Bank Conflict: 注意 Smem 的 stride 问题，使用 Padding 或 Swizzling 解决。
  - Swizzle:则在不浪费 Shared Memory 空间（无 Padding）的前提下，完美解决了 Bank Conflict，同时保证了向量化指令所需的地址对齐，是追求极致性能的终极方案。
    - swizzle 的异或代码初看可能觉得有点莫名其妙，但作为高级优化的基础技巧，值得细细理解
