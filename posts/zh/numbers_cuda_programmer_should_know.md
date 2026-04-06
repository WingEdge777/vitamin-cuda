# CUDA 开发者应该熟悉的数

这是一篇为 CUDA 开发者准备的博客，旨在总结 CUDA 编程中至关重要的硬件参数和延迟数据。

## 0. 序

在高性能计算（HPC）和深度学习领域，写出“能跑”的 CUDA 代码并不难，但要写出“极致性能”的代码，则需要对底层硬件有深刻的理解。

就像 Jeff Dean 曾经列出的“每个程序员都应该知道的延迟数字”一样，GPU 编程也有属于它的黄金数字。忽略它们，你的 GPU 可能只发挥了 5% 的功力；掌握它们，你才能真正榨干显卡的每一滴算力。

本文将带你梳理那些影响 CUDA 性能的关键常数与量级。

## 1. 核心执行单元：Warp Size = 32

这是 CUDA 编程中最著名的数字。Warp（线程束）是 GPU 执行指令的最小基本单位。

- 含义： SM（流多处理器）一次调度 32 个线程执行相同的指令（SIMT - 单指令多线程）。
- 性能启示：
  - 分支发散 (Branch Divergence)：如果一个 Warp 内的 32 个线程在同一时刻执行不同控制流路径，硬件仍以 SIMT 方式执行：各路径被谓词/掩码串行走完，活跃线程数随路径切换而变，吞吐按“有效并发”打折。Volta 及之后的独立线程调度主要改善 *within-warp* 的同步与收敛语义，并降低部分场景下因 `__syncwarp()` 等带来的额外约束，但 warp 内仍是一条指令广播给 32 线程——分歧依然有成本，别指望“独立调度=分歧免费”。
  - 内存合并 (Coalescing)： 只有当一个 Warp 内的线程访问连续对齐的内存地址时，才能实现最佳的内存吞吐量。
  - Active Masks： 在使用 Warp 级原语（如 __shfl_sync）时，你需要意识到掩码通常是 32 位的。
  - 尾部效应： 如果你的总线程数不能被 32 整除，最后的一个 Warp 会有部分线程处于非活跃状态，但仍会占用硬件资源。

潜台词：若某次 grid/block 里活跃线程总数很少，不仅难以占满 SM，还会出现未填满的尾部 Warp（部分 lane 不参与实际计算却仍随 Warp 调度），更容易暴露访存与指令延迟。更实用的说法是：尽量保证有足够多的 Warp 在跑，并让问题规模与线程组织方式能把 SIMD 宽度“吃满”。（并不是说业务上永远不能 launch 非 32 倍数的线程数。）

## 2. 内存层级与延迟 (Latencies)

理解内存延迟是优化的核心。GPU 是吞吐量导向的设备，旨在通过大量的线程切换来掩盖延迟，但延迟本身依然存在物理限制。
以下给出 Ampere/Hopper 量级上的数量级（不同 SKU、时钟、是否命中缓存、访存模式都会显著改变实测周期；勿当精确规格表背诵）。

| 存储类型 | GPU 时钟周期 (Cycles) | 物理位置 | 备注 |
| ---- | --- | ---- | ---- |
| Registers （寄存器） | 0 ~ 1 | SM 内部 | 最快，但数量有限，还需警惕指令间的读后写（RAW）延迟（常需 20+ 周期|
| Shared Memory/L1 Cache | ~20 ~ 30 | SM 内部 | 用户可控的高速缓存，需注意 Bank Conflict |
| L2 Cache | ~200 | GPU 全局共享 | 芯片上最后一道防线 |
| Global Memory | ~400 ~ 800+ | HBM/GDDR | 主要瓶颈所在 |

关键数字：

- 32 Bytes：在当代 NVIDIA GPU 上，L2 常见以 32-byte sector 为寻址/传输的较细粒度；各级缓存的 line 长度与标签粒度因架构与层级而异，不必死记“一律 128-byte line = 4×sector”，但离散访存往往以 sector 为单位拉取，因此合并访问、对齐与局部性仍然极其重要。
- 128 Bytes：一个 Warp 对连续 `float` 做向量化/合并加载时，常对应 32×4B = 128B 的合并访存模式；硬件侧可能表现为若干 32B sector 的高效组合，而不是“总是一笔 128B 原子事务”。
- 100+：为了尽量掩盖全局内存 cache miss 的数百周期延迟，往往需要大量可并发隐藏的 warp（具体要多少取决于访存强度、指令级并行、occupancy 与架构细节；“每 SM 上百 warp”这类口号不如用 profiler（如 Nsight Compute）看 stall 原因来得靠谱）。

## 3. Shared Memory Banks：32 路与 4 字节

共享内存（Shared Memory）在常见 32-bit 访存（如 `float`）下，可理解为被划分为 32 个 bank，每个 bank 通常以 4 字节宽交错映射地址（具体 bank 公式以《CUDA C Programming Guide》对你目标 compute capability 的说明为准；双宽/某些特殊模式会改变冲突画像）。

- Bank Conflict（存储体冲突）：同一 warp 内若有多路访问落在同一 bank 的不同地址上，硬件需把访问拆成多次串行服务（体现为额外延迟与有效带宽下降）。
- 最坏情况：32-way conflict 表示理想情况下一次 warp 访存被劈成约 32 段才能服务完（周期数会大于 32，还需结合访存指令自身延迟与架构微结构；下面的除法仍是有效的定性心智模型）。

Effective Bandwidth = Peak Bandwidth / Conflict Degree

## 4. 数据传输：PCIe vs. NVLink

永远记住：数据搬运是性能杀手。任何 Host (CPU) 与 Device (GPU) 之间的数据传输都极其昂贵。

- PCIe Gen4 x16: 单向理论有效带宽约 31~32 GB/s（16 GT/s ×16 lane，再扣 128b/130b 编码）；市场上常把双向合计口语化成约 64 GB/s，请勿与“单向 64 GB/s”混淆。
  - 实测 goodput 常低于理论峰值：除协议开销外，还与是否 pinned (`cudaHostRegister`/`cudaMallocHost`)、是否异步流水线、拷贝粒度与 TLP、Root Complex/CPU 拓扑等有关；单向 25~30 GB/s 在不少桌面/工作站平台上比较常见，偏低时要先排查链路是否被折成 x8、是否走 PCH、以及拷贝路径。
- PCIe Gen5 x16: 单向理论有效带宽约 63~64 GB/s；双向合计常称约 128 GB/s（同上，别与单向混淆）。
- NVLink（以 H100 系为例）：公开资料里常见 ~900 GB/s 量级，多指机型/拓扑下多链路聚合的 GPU-GPU 互联带宽峰值；具体到单条链路/单机内可路由路径，请以对应白皮书与系统规格为准。
- GPU 显存带宽（H100 HBM3 量级）：峰值约 3.35 TB/s（3350 GB/s）一档（仍与 SKU、散热与实测 sustained 有别）。

性能启示：把 PCIe 有效吞吐和 HBM 峰值带宽直接相除，得到的是数量级（常见约百分之几）；优化时更应关心是否频繁 host/device 往返、是否可融合/驻留显存、是否可 overlap 计算与拷贝。
结论：尽可能将计算留在 GPU 上；即便某些步骤 GPU 并不绝对擅长，也常被 PCIe 往返更划算（仍要结合精度、库与工程约束综合判断）。

## 5. Kernel Launch Overhead：常见「几微秒」量级

启动一个 Kernel 并不是免费的：主机侧把 work 提交给 GPU 往往要 ~几 µs（常见经验值 ~3–10 µs 都存在，冷启动、驱动/电源状态、同步方式、以及是否在同一 stream 已连续提交都会让你测到的数字差很多；CUDA Graph 重放可把每次提交摊薄到更低）。

- 若某个 kernel 本体只有 ~1 µs 量级甚至更短，经常出现启动与调度开销压过计算的现象。
- 对策：CUDA Graph、合并小 kernel（fusion）、减少不必要的同步、以及用 batch 让小任务“长得像一个大任务”。

## 总结：优化清单

在编写下一行 CUDA 代码前，请问自己：

- 我的 grid/block 是否提供了足够的 warp/occupancy 来隐藏访存与指令延迟？（别只盯着“32”，要看活跃 warp 数与寄存器/shared 等资源约束。）
- 我的全局访存是否合并/共线？（warp 常见合并模式常与 32×4B=128B 及其 32B sector 组合相关；以 Nsight Compute 的 memory 指标为准。）
- 我是否避免了 Bank Conflict？(32 banks)
- 我是否在用 PCIe 传输小数据？(Bandwidth limitations)
- 我的 Kernel 足够大吗？(Launch overhead)
