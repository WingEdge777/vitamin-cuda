# 深度解读 DeviceQuery：理解你的 GPU 硬件属性

## 0. 序

读本文前最好先有基本的 CUDA 编程基础，对 GPU 的计算能力、内存、cache、warp、block、gride 等概念有所了解。

## 1. Device Query 输出

以本人的 **RTX 5060** 显卡为例。

```bash
cd samples/deviceQuery && bash run.sh
```

### 1.1 输出

```yaml
:: 1 available devices

  CUDA Driver Version / Runtime Version          13.1 / 12.9
  CUDA Capability Major/Minor version number:    12.0
  Total amount of global memory:                 8151 MBytes (8546484224 bytes)
  (026) Multiprocessors, (128) CUDA Cores/MP:    3328 CUDA Cores
  GPU Max Clock rate:                            1455 MHz (1.46 GHz)
  Memory Clock rate:                             12001 MHz
  Memory Bus Width:                              128-bit
  L2 Cache Size:                                 33554432 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        102400 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      No
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::CUDASetDevice() with device simultaneously) >
```

## 2. 详细解读

这是一张 RTX 5060，虽然参数都在上面列着，但对于开发者来说，更重要的是理解这些数字背后的性能瓶颈和优化方向。

### 2.1 基本架构

- CUDA 驱动版本 / 运行时版本：13.1 / 12.9
- CUDA 算力兼容版本号 (Compute Capability)：12.0 (Blackwell 架构）
  - 架构决定了支持的指令集。通常高版本架构意味着更好的低精度计算性能，Blackwell 架构原生支持 FP4, FP8, BF16, INT8 等低精度计算，非常适合大模型推理量化。

### 2.2 核心算力和并发能力

- 流处理器 (SM) 数量：26 个
  - 每个流处理器的 CUDA 核心数：128 个
    - 即每个 SM 真实并行的线程数为 128
  - 总 CUDA 核心数：3328 个 (26 * 128)
    - 这意味着硬件层面同时真实执行的线程数最多 3328 个
  - 编程建议：虽然物理核心只有 3k 多，但我们由 GPU 的设计哲学（掩盖延迟）决定了我们必须发射远超这个数量的线程。
  - Grid 设置：Kernel 的线程组 (Block) 数量最好是 SM 数 (26) 的整数倍，且最好有几百个以上的 Block，这样调度器才能充分轮转 Block 以隐藏延迟。
- 每个 SM 最多驻留线程数：1536
- 每个 block 最大线程数：1024
  - Occupancy （占用率） 计算：
    - 如果设为 1024 线程/Block：一个 SM 只能跑 1 个 Block（1024 < 1536，但跑 2 个就超了）。利用率 = 1024/1536 ≈ 66%
    - 如果设为 512 线程/Block：一个 SM 可以跑 3 个 Block。利用率 = 1536/1536 = 100%
  - 甜点位 (Sweet Spot)：通常 128 或 256 是比较通用的 Block Size，容易凑出高利用率。这个大小能比较平衡地分配寄存器/共享内存资源。当然，在极致优化时需要具体分析（Occupancy Calculator 算一下）
  - 整卡活跃线程上限：1536 * 26 = 39,936 个

### 2.3 时钟频率

- GPU 最大频率：1455 MHz (1.46 GHz)
  - 约 0.5ns 执行一个最简单的指令 （如 FADD)。
  - 对比 CPU：
    - 我的 Ultra 9 285H 主频高达 2.9GHz+。单看频率，再加上 CPU 的分支预测和乱序执行能力等等，GPU 跑串行逻辑就是“垃圾中的战斗机”。
  - 设计哲学：GPU 讲究**“三个臭皮匠顶个诸葛亮”**。它不追求单线程极速，而是追求几万个线程 (39,936 个） 同时推进。
  - 编程建议：绝对不要在 Kernel 里写复杂的串行逻辑或深层 if/else 嵌套。
- 显存频率：12001 MHz
  - 这个频率是等效频率，实际频率计算比较复杂（涉及 Command Clock, Write Clock, Double Data Rate 等），我们开发时只需关注它计算出的带宽吞吐。
  - Latency vs Throughput：这个频率高不代表延迟低，单次显存读取的延迟可能高达几百个 GPU 时钟周期，所以 CUDA 程序往往都是 Memory Bound（尤其这张卡），优化的核心永远是掩盖这些延迟。

### 2.4 内存子系统（性能瓶颈之源）

- 总显存大小：8GB
- 显存位宽：128 bits
  - 位宽很小了，等效频率也不高，作为对比参考（A100 ：5120 bits=128*40），可以计算出显存带宽 = 12001*2*128/8 = 384000MB /s ≈ 384G/s，我们的 kernel 大概率是带宽瓶颈，而如果 kernel 在处理大规模数据里，带宽吞吐达不到这个量级，说明还有提升空间（更何况还有 L2 cache 提速）
  - 现状：128-bit 位宽非常窄，这意味这块卡的 Kernel 极大概率是带宽瓶颈。
  - 合并访问 (Coalescing)：虽然物理显存有 Burst Size，但从 CUDA 核心视角看，Global Memory 访问的最小粒度是 32 字节 (Sector)。
    - 正面案例：32 个线程正好读取连续的 128 字节 （如 float)，合并为 4 个 Sector 事务。
    - 反面案例：如果只读 1 个字节，或者跨步读取，总线依然要搬运整个 32 字节 Sector，导致带宽浪费。
  - 编程建议：必须保障相邻线程读写地址的连续性，即程序的空间局部性；尽量使用向量化读写 (float4) 来减少指令发射量（提高指令并行度）。
- L2 缓存大小：32MB
  - 重要性：L2 Cache 也是以 32B Sector 为管理单位（通常 Cache Line 为 128B）。
  - 空间局部性：L2 是全卡共享的，且速度比显存快得多。编程时应利用**“空间局部性 (Spatial Locality)”**，让同一个 Warp 的线程读取相邻数据，最大化 L2 命中率。对于 5060 这类显存位宽小的卡，32MB 的大 L2 是救命稻草。
- 总的常量内存：65536 bytes (64KB)
  - 常量内存是只读的，所有线程共享，用于存储常量数据，比如权重等。常量内存的访问速度比全局内存快，因为常量内存有专门的缓存。

### 2.5 资源限制和 Occupancy（占用率）

- 每个 SM 最大共享内存 (Shared Memory)：100 KB
- 每个 Block 共享内存限制：48 KB
  - 坑点：如果你一个 Block 申请了 48KB Shared Mem，那么一个 SM 最多只能跑 2 个 Block (48*2 < 100)，这可能导致 SM 没跑满，降低 Occupancy。
- 每个 Block 寄存器文件 (Register File)：65536 个
  - 除了共享内存限制，每个线程可以使用的寄存器数量也是有限的，如果一个线程使用了过多的寄存器，会导致 活跃线程数/block 数减少，影响 occupancy
  - 寄存器溢出 (Register Spill)：除了 Shared Memory，寄存器也是稀缺资源。如果单线程使用的寄存器过多，会导致 SM 能并行的 Block 数量减少
  - 严重后果：如果寄存器彻底不够用，编译器会把变量“溢出”到 Local Memory。注意！Local Memory 物理上是显存，速度极慢，会严重拖垮流水线
- Bank Conflict：共享内存被划分为 32 个 Bank。这已经讲烂了，总而言之就是一个 warp 线程访问共享内存时，多个线程不能同时访问同一个 bank 的不同地址数据，否则会变成串行的多次访问，共享内存的 bank id 归属是每 4 字节 (32bits) 顺序归属的，即连续的 0...31，0...31 这样
  - 建议：最简单的方法是让相邻线程访问相邻地址 (tid 对应 data[tid])，这样天然无冲突。当然有时为了复杂的任务分发而进行下标变换会进行交错访问，那就要很小心地避免冲突了

### 2.6 调度和物理限制

- warp size：32
  - warp 是 gpu 调度的最小单位，每个 warp 包含 32 个线程，这些线程必须同时执行相同的指令，这是 gpu 并行计算的基础
    - 指定的线程组 block 大小为比如 256 个，他们是每 32 个一组一组的被调度到 SM 上运行
  - Warp Divergence：这个也要讲烂了，因为是 SIMT model，如果有 if/else 分支，那么两个分支都串行执行，只是在对应阶段会屏蔽条件外的线程。
- block 的最大 shape (x,y,z): (1024, 1024, 64)
  - 是的，block 的纬度是有大小限制的，毕竟上面也写了，每个 block 最多 1024 个线程；
  - 多维只是为了在特别情况下，索引数据方便多维读取和理解
- grid 的最大 shape (x,y,z): (2147483647, 65535, 65535)
  - 一般来说，可以放心无脑地用单个数字作为 grid（爆 int 除外），除非数据特别适合多维索引，那么可以改用多维 grid
- 最大内存 pitch：2147483647 bytes
  - 定义二维显存时，单行内存最大 stride 不能爆 int
- 纹理内存地址对齐：512 bytes
  - 纹理内存地址必须对齐到 512 字节，这是为了提高纹理内存的访问效率
- Async Copy：
  - 支持 Copy Engine 和 Compute Engine 并行（双流水线），这是实现计算与传输重叠 (Overlap) 的硬件基础。
- 剩下的就没什么好说的了

## 3. 最终建议小结

- Block Size: 常用 128 或 256，且 Grid Size 至少要是 SM 数（26）的几倍甚至几十倍。
- 访存：
  - 访存铁律：
    - Coalesced: 必须连续访问。
    - Vectorized: 能用 float4 绝不用 float。
- 延迟隐藏：不要怕线程多，利用海量线程切换来掩盖内存延迟。
- 资源管理：盯着 Shared Memory 和 Register 用量，防止 Occupancy 暴跌或寄存器溢出到显存。

以上是自己对 GPU 架构和 CUDA 编程的一些理解，希望对大家有所帮助。欢迎批评指正！
