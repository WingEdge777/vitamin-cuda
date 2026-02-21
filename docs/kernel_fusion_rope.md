# [CUDA 优化实战] RoPE - 手写算子的作用之 kernel fusion：减少访存、摊薄启动开销的优化技巧

本文直接聚焦一个核心命题：为什么“手写算子（hand-written operator）”与“内核融合（kernel fusion）”能够带来大幅度的性能提升？本文基于 RoPE kernel 的经典的 pytorch naive 实现（neo-x style），pytorch cos/sin 表cache实现、单个 CUDA kernel 实现说明工程化结论和优化点。

简短结论：

- 单纯在 PyTorch 层做表缓存（cos/sin）可以 F 减掉一部分开销
- 但手写内核有核心的、不可替代的提速，分别来自于的
  - 减少对特征向量读取次数
  - 带框瓶颈，以算代读，降低空间复杂度，提高 kernel 执行速度
  - 读/算/写操作融合在一个 kernel 内部，减少 kernel lauch 开销（一般 2~5 ns，但现在都不提这个了，因为 CUDA graph 基本解决了这个问题且CUDA graph已经成为推理框架标配）

## 0. 分析 pytorch naive 实现

上代码
```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed
```

框架层实现（例如 PyTorch 的按元素实现）追求通用性，会将操作拆为许多小步，产生大量小 kernel 与频繁的显存往返。相比之下，手写算子可以在实现层面对数据布局、线程映射、对齐与载入/存储粒度进行逐字节优化：

- 把多个 32-bit 读合并成一次 128-bit 读写（`float4`），显著减少全局内存事务；
- 在同一个 kernel 内完成读—算—写，避免中间结果往返显存并摊薄 kernel 启动开销；
- 利用寄存器保存中间值，减少对 L2/DRAM 的依赖，从而把瓶颈从内存迁移到算术单元。

这些能力是单靠 PyTorch 层缓存三角表无法替代的，缓存仅解决三角函数计算的重复问题，而无法改变内存访问粒度与 kernel 调度成本。

## 1. RoPE 的性能热点（简要剖析）

- 访存次数：每个元素的 load/store 会产生大量小事务；
- kernel 启动：大量小 kernel 会被反复调度，固定延迟累加显著；
- 对齐与合并访问：未对齐或非合并的访问会导致低效的 DRAM/缓存利用率。

工程目标就是用手写内核直接处理这三项：把多个元素合并为一次 load/store，保持连续与对齐；把多步逻辑融合为一个 kernel；在内核中用寄存器缓存中间结果。

## 2. 手写算子的三大杠杆（工程实践）

1) 向量化 load/store（128-bit / `float4`）

- 把连续分量用 `float4` 一次性读取：4 次访问 → 1 次访问；在寄存器中完成 4 个旋转后一次性写回。
- 要点：保证数据对齐（16 字节），处理好 tail 情况。

2) 内核融合（Kernel Fusion）

- 把读取、旋转、写回放在同一个 kernel，避免中间写回显存；
- 减少 kernel 启动次数，摊薄每次 launch 的固定延迟。

3) 寄存器与线程组织优化

- 在线程层面做局部复用：在寄存器里保存已加载向量并复用，减少对 L2/DRAM 的重复访问；
- 选择合理的 block/warp 大小以平衡 occupancy 与寄存器压力。

把这三者组合起来，就是本仓库 `rope` / `rope_fp32x4` 手写内核的核心策略。

## 3. 关于 cos/sin 缓存的说明（重要）

仓库的 `test.py` 包含一个在 PyTorch 层做的 `rope_with_sin_cos_cache`，它在 host/PyTorch 端预计算並缓存 cos/sin 表以减少三角函数重复计算。需要说明两点：

- 这是对 PyTorch naive 实现的优化（减少三角函数开销），而不是把缓存逻辑实现为单独的 CUDA 算子来替代手写内核；
- 即便在 PyTorch 层做了缓存，性能仍不及手写内核带来的整体提升——因为缓存无法改变内存访问粒度、对齐或 kernel-launch 开销。

因此请把 cos/sin 缓存视为 baseline 的工程优化，而不是手写内核的等价替代。

## 4. 代码片段（概念演示）

示例（PyTorch 层构建 cos/sin，仅作为 baseline）：

```py
INV_FREQ = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
freqs = torch.outer(position_ids, INV_FREQ)
cos = torch.cos(freqs)
sin = torch.sin(freqs)
```

核心手写内核的访存与融合思路（伪码）：

```cpp
// 在单个线程中一次性读取 4 个相邻分量
float4 x = LDST128BITS(a[idx]);
float4 y = LDST128BITS(a[idx + head_dim/2]);
// 在寄存器内完成 4 次旋转
LDST128BITS(b[idx]) = x_new;
LDST128BITS(b[idx + head_dim/2]) = y_new;
```

注意关注：对齐、tail 处理、寄存器占用（避免过高导致 occupancy 下降）。

## 5. 仓库基准摘要（要点）

- 被测项：`torch`（naive）、`torch.rope_with_sin_cos_cache`（PyTorch 层缓存）、`rope`（hand-written kernel）、`rope_fp32x4`（vectorized hand-written）。
- 观测：手写内核（特别是 `rope_fp32x4`）在多数配置下明显优于 PyTorch 层实现（含缓存），因为它改写了内存访问粒度並摊薄了调度开销。

典型数值（节选）：

- PyTorch naive: ~0.25–0.56 ms
- `rope` (hand-written fp32 kernel): ~0.01–0.03 ms
- `rope_fp32x4` (hand-written vectorized): ~0.008–0.02 ms

结论：缓存三角表能减少一部分开销，但手写内核带来的向量化与融合是决定性的性能因子。

## 6. 实践建议清单（Actionable）

1. 优先写手写内核来控制访存粒度与 kernel 结构，而不是先做 PyTorch 层微优化；
2. 用 `float4` 做向量化读写并保证对齐；
3. 把读—算—写融合到一个内核，避免显存往返；
4. 测试寄存器使用与 occupancy（权衡展开与并行度）；
5. 用 `test.py` 的 `diff_check` 校验每次修改的数值正确性。

## 7. 复现命令

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python kernels/rope/test.py
```

## 8. 拓展思路

- 把 RoPE 与后续的 QK 乘积、缩放或 softmax 做更深层次融合，进一步避免中间张量写回；
- 在支持的架构上尝试 `cp.async`（异步拷贝）与 swizzle / tile layout，以兼顾对齐與 bank-conflict 的消除；
- 引入混合精度（FP16/BF16），配合向量化，继续压缩访存带宽需求。

## 小结

手写算子的价值不在于替代高层缓存策略，而在于它能从实现层面重构数据通路：改变访存粒度、降低 kernel 调度次数、并在内核内部高效复用寄存器与缓存。这些是 PyTorch 层缓存无法替代的优化路径，也是 `rope` / `rope_fp32x4` 能显著胜出的根本原因。

参考实现：
- `kernels/rope/rope_neox.cu`
- `kernels/rope/test.py`
