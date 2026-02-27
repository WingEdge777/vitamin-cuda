# [CUDA 优化实战] RoPE - 手写算子的作用之 kernel fusion：减少访存次数、减少启动开销的优化技巧

现在 jit 优化，ai 编译器进化的越来越快，经常看到人说没必要手写算子了。

本文直接聚焦一个核心命题：为什么“手写算子（hand-written operator）”与“内核融合（kernel fusion）”能够带来大幅度的性能提升？本文基于 RoPE kernel 的经典的 pytorch naive 实现（neo-x style），pytorch cos/sin 表 cache 实现 以及 单个 CUDA kernel 实现说明手写算子的优势和结论。

完整代码可参见链接：

简短结论：

- 单纯在 PyTorch 层做表缓存（cos/sin）可以减掉一部分开销
- 但手写内核有核心的、不可替代的优化点，分别来自于的
  - 减少对特征 q 的读取次数（访存是最慢的，flash attn 的优化核心也是减少访存）
  - 带宽瓶颈，以算代读，降低空间复杂度，提高 kernel 执行速度
  - 读/算/写操作融合在一个 kernel 内部，减少 kernel lauch 开销（一般 2~5 ns，但现在都不提这个了，因为 CUDA graph 基本解决了这个问题且 CUDA graph 已经成为推理框架标配）

## 0. 分析 pytorch naive 实现

上代码

```python

def compute_default_rope_parameters(head_dim):
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2).float().cuda() / head_dim)
    )  # 64
    return inv_freq

# 预处理好 freqs
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

# neo-x style rope, 假设单头（多头算进 bs 纬度就可以了）
#@torch.compile()
def rope(q):  # q shape: [bs, seqlen, head_dim]
    inv_freq = compute_default_rope_parameters(q.shape[-1])
    position_ids = torch.arange(q.shape[1], device=q.device).float()

    # [seq_len] outer [dim/2] -> [seq_len, dim/2]
    freqs = torch.outer(position_ids, inv_freq)

    # [seq_len, dim/2] -> [seq_len, dim]
    freqs = torch.cat([freqs, freqs], dim=-1)

    cos, sin = torch.cos(freqs), torch.sin(freqs)
    cos = COS[q.shape[-1]][:q.shape[1], :q.shape[-1]]
    sin = SIN[q.shape[-1]][:q.shape[1], :q.shape[-1]]

    return apply_rotary_pos_emb(q, cos, sin)

#@torch.compile()
def rope_with_sin_cos_cache(q):  # q shape: [bs, seqlen, head_dim]
    cos = COS[q.shape[-1]][:q.shape[1], :q.shape[-1]]
    sin = SIN[q.shape[-1]][:q.shape[1], :q.shape[-1]]

    return apply_rotary_pos_emb(q, cos, sin)
```

在 pytorch fp32 实现中，即使做了 COS/SIN 表 cache，q 的最后一维度前后一半要分别与另一半相乘，所以其读写数据量就有 `bs * seq_len * head_dim * 2 * 4` bytes， 加上 COS/SIN 的读取又有两倍，还有临时空间的读写

- 因此共有读取量 `bs * seq_len * head_dim * 4` bytes * 11（q 的两次读取，SIN/COS 表读取，q_out 写回；临时空间有rotate_half(q)、q*cos、rotate_half(q)*sin 的读写）
  - 当然这是我的最原始的粗略估计，实际pytorch默认可能有什么优化策略导致读写量与我描述的不符

手写算子可以有以下优势：

- One pass, 只读一遍 q，以算代读避免其他冗余读写（代价是 sin/cos 的重复计算）
- 在同一个 kernel 内完成读—算—写，避免中间结果往返显存并摊薄 kernel 启动开销
- 因此带宽读写总量仅仅为 `bs * seq_len * head_dim * 4`bytes * 2, 降低了 **~82%的读写量**

这些能力是单靠 PyTorch 层缓存三角表无法替代的，缓存仅解决三角函数计算的重复问题，而无法改变内存访问次数。

## 1. 手写算子的工程实践

- 向量化 load/store（128-bit / `float4`），其实不管是 cpu 还是 gpu，访存都太慢了（相对计算单元来说），因此时时刻刻都要记着尽量加速访存读写
  - 把连续数据用 `float4` 一次性读取：4 次访问 → 1 次访问；在寄存器中完成 4 个旋转后一次性写回。
  - 现在 llm 模型 head_size 肯定是 4 的倍数无需担心对齐问题

- 内核融合（Kernel Fusion）
  - 把读取、旋转、写回放在同一个 kernel，避免中间写回显存；
  - 减少 kernel 启动次数，摊薄每次 launch 的固定延迟。（可以不提了）

两点结合一下，完成以下代码：

```c++
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])
// rope neox float4 版
// a[bs, seq_len, head_dim]
__global__ void rope_fp32x4_kernel(float *a, float *b, int seq_len, int head_dim) {
    int pos = blockIdx.x * 4;
    int tid = threadIdx.x;
    pos += tid * 4 / (head_dim >> 1);
    int f_idx = tid * 4 % (head_dim >> 1);
    int idx = blockIdx.y * (seq_len * head_dim) + pos * head_dim + f_idx;
    // 合并读取
    float4 x = LDST128BITS(a[idx]);
    float4 y = LDST128BITS(a[idx + (head_dim >> 1)]);

    float inv_freq[4], c[4], s[4];
    // 计算旋转角度
#pragma unroll
    for (int i = 0; i < 4; i++) {
        inv_freq[i] = 1.f / __powf(theta, (2.0f * (f_idx + i)) / head_dim);
        __sincosf(pos * inv_freq[i], &s[i], &c[i]);
    }
    // 旋转
    float4 x_new, y_new;
    x_new.x = x.x * c[0] - y.x * s[0];
    x_new.y = x.y * c[1] - y.y * s[1];
    x_new.z = x.z * c[2] - y.z * s[2];
    x_new.w = x.w * c[3] - y.w * s[3];
    y_new.x = y.x * c[0] + x.x * s[0];
    y_new.y = y.y * c[1] + x.y * s[1];
    y_new.z = y.z * c[2] + x.z * s[2];
    y_new.w = y.w * c[3] + x.w * s[3];

    // 合并写回
    LDST128BITS(b[idx]) = x_new;
    LDST128BITS(b[idx + head_dim / 2]) = y_new;
}
```

### 1.1 关于 cos/sin 缓存的说明

仓库的 `test.py` 包含一个在 PyTorch 层做的 `rope_with_sin_cos_cache`，它在 host/PyTorch 端预计算並缓存 cos/sin 表以减少三角函数重复计算。需要说明两点：

- 这是对 PyTorch naive 实现的优化（减少三角函数开销），而不是把缓存逻辑实现为单独的 CUDA 算子来替代手写内核；
- 即便在 PyTorch 层做了缓存，性能仍不及手写内核带来的整体提升——因为缓存无法改变内存访问粒度、对齐或 kernel-launch 开销。

因此请把 cos/sin 缓存视为 baseline 的工程优化，而不是手写内核的等价替代。

## 2. benchmark 结果与分析

这里做了关于 torch.compile 开启与否两个版本的对比

- 不开启

```yaml
bs: 128, n: 8192, m: 128
torch                          mean time: 86.233111 ms
torch.rope_with_sin_cos_cache  mean time: 56.642383 ms
rope                           mean time: 3.187057 ms
rope_fp32x4                    mean time: 3.110710 ms
```

- 开启

```yaml
bs: 128, n: 8192, m: 128
torch                          mean time: 5.242156 ms
torch.rope_with_sin_cos_cache  mean time: 4.811796 ms
rope                           mean time: 3.331872 ms
rope_fp32x4                    mean time: 3.306061 ms
```

可以看到，torch.compile 的效果的确很显著，现在的框架 jit 优化+autotune 的效果拔群，但在一些需要特殊优化的场景下，仍然需要我们自己来写高性能的 kernel。比如这里，即使在 torch.compile 开启的情况下，手写 kernel 仍然有优势。

计算手写算子带宽吞吐为  128*8192*128*4*2 / 3.306061 * 1e3 / 1e9 = 325 GB/s,我的显卡（RTX 5060 移动版）理论显存极限物理带宽为 384GB/s, 达到85%的理论带宽利用率（由于是笔记本测试，无法排除桌面等应用程序的干扰）。

手写的 rope kernel 通过向量化读取、以算代读等手段最大化提高带宽利用效率，性能还是领先于 torch.compile 优化的 naive 版本。而且本文仅仅是以 rope 为例说明手写算子的一些优势，并未对 rope 的实现做更深入的优化和分析，cos/sin 可以提前算好通过常量/纹理内存进行加速，以算代写中寄存器计算的结果可以循环多次复用等等，实际上 rope 还常常和线性层或和 attention 融合计算，不过就不多讨论了。此外对于一些自定义的 op，或者复杂度更高的 op，目前的自动优化措施终究无法达到手写 kernel 的极致水平。

以上，共勉。
