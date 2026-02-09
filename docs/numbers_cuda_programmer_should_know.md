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
  - 分支发散 (Branch Divergence)： 如果一个 Warp 内的 32 个线程走向了不同的 if-else 分支，硬件必须串行执行这些分支，导致性能急剧下降。
  - 内存合并 (Coalescing)： 只有当一个 Warp 内的线程访问连续对齐的内存地址时，才能实现最佳的内存吞吐量。
  - Active Masks： 在使用 Warp 级原语（如 __shfl_sync）时，你需要意识到掩码通常是 32 位的。
  - 尾部效应： 如果你的总线程数不能被 32 整除，最后的一个 Warp 会有部分线程处于非活跃状态，但仍会占用硬件资源。

潜台词： 任何少于 32 个线程的工作负载都是对算力的浪费。

## 2. 内存层级与延迟 (Latencies)

理解内存延迟是优化的核心。GPU 是吞吐量导向的设备，旨在通过大量的线程切换来掩盖延迟，但延迟本身依然存在物理限制。
以下是基于 Ampere/Hopper 架构的典型估算值

| 存储类型 | GPU 时钟周期 (Cycles) | 物理位置 | 备注 |
| ---- | --- | ---- | ---- |
| Registers （寄存器） | 0 ~ 1 | SM 内部 | 最快，但数量有限 |
| Shared Memory/L1 Cache | ~20 ~ 30 | SM 内部 | 用户可控的高速缓存，需注意 Bank Conflict |
| L2 Cache | ~200 | GPU 全局共享 | 芯片上最后一道防线 |
| Global Memory | ~400 ~ 800+ | HBM/GDDR | 主要瓶颈所在 |

关键数字：

- 128 Bytes：全局内存事务的粒度。即便你只读取 1 个 float (4 bytes)，硬件也可能一次性拉取 128 bytes。如果 Warp 内的线程访问不连续，会导致严重的带宽浪费。
- 100+： 为了完全掩盖全局内存访问的几百个周期延迟，你需要每个 SM 至少有数百个活跃 Warp 处于“ In-flight”状态。

## 3. Shared Memory Banks：32 路与 4 字节

共享内存（Shared Memory）虽快，但它被划分为 32 个 Bank（存储体），每个 Bank 宽度为 4 字节（32-bits）。

- Bank Conflict（存储体冲突）： 当 Warp 内的多个线程试图访问同一个 Bank 的不同地址时，访问必须串行化。
- 最坏情况： 32 线程冲突（32-way conflict），原本 1 个周期的访问变成了 32 个周期。

Effective Bandwidth = Peak Bandwidth / Conflict Degree

## 4. 数据传输：PCIe vs. NVLink

永远记住：数据搬运是性能杀手。任何 Host (CPU) 与 Device (GPU) 之间的数据传输都极其昂贵

- PCIe Gen4 x16: 带宽约为 64 GB/s （双向）
- PCIe Gen5 x16: 带宽约为 128 GB/s （双向）
- NVLink (H100): 带宽可达 900 GB/s
- GPU 显存带宽 (HBM3): 高达 3,350 GB/s

性能启示： PCIe 的速度仅为 GPU 内部显存速度的 2% ~ 4%
结论： 尽可能将计算留在 GPU 上，即便某些步骤 GPU 并不擅长，也比通过 PCIe 传回 CPU 处理再传回来要快

## 5. Kernel Launch Overhead：~2-5 微秒

启动一个 Kernel 并不是免费的。CPU 通知 GPU 启动任务大约需要 ~2-5 us。

- 如果你有一个 Kernel 执行时间仅为 1 us，那么你花了 90% 的时间在“打电话通知 GPU 干活”，而不是在“干活”。
- 解决方案： 使用 CUDA Graphs 或将小 Kernel 合并（Kernel Fusion）。

## 总结：优化清单

在编写下一行 CUDA 代码前，请问自己：

- 我的 Warp 满载了吗？(32 threads)
- 我的内存访问合并了吗？(128 bytes transaction)
- 我是否避免了 Bank Conflict？(32 banks)
- 我是否在用 PCIe 传输小数据？(Bandwidth limitations)
- 我的 Kernel 足够大吗？(Launch overhead)
