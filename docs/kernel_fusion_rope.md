# [CUDA 优化实战] RoPE - 手写算子的作用之 kernel fusion：减少访存次数、减少启动开销的优化技巧

本文直接聚焦一个核心命题：为什么“手写算子（hand-written operator）”与“内核融合（kernel fusion）”能够带来大幅度的性能提升？本文基于 RoPE kernel 的经典的 pytorch naive 实现（neo-x style），pytorch cos/sin 表cache实现 以及 单个 CUDA kernel 实现说明手工优化的原理和结论。

完整代码可参见链接：

简短结论：

- 单纯在 PyTorch 层做表缓存（cos/sin）可以减掉一部分开销
- 但手写内核有核心的、不可替代的优化点，分别来自于的
  - 减少对显存读取次数（访存是最慢的，flash attn的优化核心也是减少访存）
  - 带宽瓶颈，以算代读，降低空间复杂度，提高 kernel 执行速度
  - 读/算/写操作融合在一个 kernel 内部，减少 kernel lauch 开销（一般 2~5 ns，但现在都不提这个了，因为 CUDA graph 基本解决了这个问题且CUDA graph已经成为推理框架标配）

## 0. 分析 pytorch naive 实现

上代码

```python

def compute_default_rope_parameters(head_dim):
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2).float().cuda() / head_dim)
    )  # 64
    return inv_freq


INV_FREQS = {
    256: compute_default_rope_parameters(256),
    128: compute_default_rope_parameters(128),
}

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed

# neo-x stype rope, single head single batch
def rope(q):  # q shape: [seqlen, head_dim]
    inv_freq = compute_default_rope_parameters(q.shape[1])
    position_ids = torch.arange(q.shape[0], device=q.device).float()

    # [seq_len] outer [dim/2] -> [seq_len, dim/2]
    freqs = torch.outer(position_ids, inv_freq)

    # [seq_len, dim/2] -> [seq_len, dim]
    freqs = torch.cat([freqs, freqs], dim=-1)

    cos, sin = torch.cos(freqs), torch.sin(freqs)
    cos = COS[q.shape[1]][:q.shape[0], :q.shape[1]]
    sin = SIN[q.shape[1]][:q.shape[0], :q.shape[1]]

    return apply_rotary_pos_emb(q, cos, sin)

def rope_with_sin_cos_cache(q):  # q shape: [seqlen, head_dim]
    # inv_freq = compute_default_rope_parameters(q.shape[1])
    # position_ids = torch.arange(q.shape[0], device=q.device).float()

    # # [seq_len] outer [dim/2] -> [seq_len, dim/2]
    # freqs = torch.outer(position_ids, inv_freq)

    # # [seq_len, dim/2] -> [seq_len, dim]
    # freqs = torch.cat([freqs, freqs], dim=-1)

    # cos, sin = torch.cos(freqs), torch.sin(freqs)
    cos = COS[q.shape[1]][:q.shape[0], :q.shape[1]]
    sin = SIN[q.shape[1]][:q.shape[0], :q.shape[1]]

    return apply_rotary_pos_emb(q, cos, sin)
```

在pytorch实现中，即使做了COS/SIN表cache，q的第二纬度前后一半要分别与另一半相乘，所以其读写数据量就有 `seq_len * head_dim` * 2， 加上COS/SIN的读取又有两倍

- 因此共有读取量 `seq_len * head_dim` *2* 4（q的两次读取，SIN/COS表读取，临时空间读+写

手写算子有以下优势

- 只读一遍 q，以算代读避免其他冗余读写（代价是sin/cos的重复计算）
- 在同一个 kernel 内完成读—算—写，避免中间结果往返显存并摊薄 kernel 启动开销

这些能力是单靠 PyTorch 层缓存三角表无法替代的，缓存仅解决三角函数计算的重复问题，而无法改变内存访问次数。

## 1. RoPE 的性能热点（简要剖析）

- 访存次数：每个元素的 load/store 会产生大量小事务；
- 对齐与合并访问：未对齐或非合并的访问会导致低效的 DRAM/缓存利用率。

工程目标就是用手写内核直接处理这三项：把多个元素合并为一次 load/store，保持连续与对齐；把多步逻辑融合为一个 kernel；在内核中用寄存器缓存中间结果。

## 2. 手写算子的三大杠杆（工程实践）

1) 向量化 load/store（128-bit / `float4`）

- 把连续分量用 `float4` 一次性读取：4 次访问 → 1 次访问；在寄存器中完成 4 个旋转后一次性写回。
- 现在llm模型head_size肯定是4的倍数无需担心

2) 内核融合（Kernel Fusion）

- 把读取、旋转、写回放在同一个 kernel，避免中间写回显存；
- 减少 kernel 启动次数，摊薄每次 launch 的固定延迟。(可以不提了)


## 3. 关于 cos/sin 缓存的说明（重要）

仓库的 `test.py` 包含一个在 PyTorch 层做的 `rope_with_sin_cos_cache`，它在 host/PyTorch 端预计算並缓存 cos/sin 表以减少三角函数重复计算。需要说明两点：

- 这是对 PyTorch naive 实现的优化（减少三角函数开销），而不是把缓存逻辑实现为单独的 CUDA 算子来替代手写内核；
- 即便在 PyTorch 层做了缓存，性能仍不及手写内核带来的整体提升——因为缓存无法改变内存访问粒度、对齐或 kernel-launch 开销。

因此请把 cos/sin 缓存视为 baseline 的工程优化，而不是手写内核的等价替代。

## 5. benchmark 结果与分析

```yaml
bs: 128, n: 8192, m: 128
torch                          mean time: 86.233111 ms
torch.rope_with_sin_cos_cache  mean time: 56.642383 ms
rope                           mean time: 3.187057 ms
rope_fp32x4                    mean time: 3.110710 ms
```

手写算子的单个kernel通过多种手段最大化提高带宽利用效率，以算代读
