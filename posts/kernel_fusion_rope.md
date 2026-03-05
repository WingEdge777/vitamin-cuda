# [CUDA 优化实战] RoPE - 手写算子的作用之 kernel fusion：减少访存次数、减少启动开销的优化技巧

## 背景

现在 AI 编译器进化得越来越快，PyTorch 的 torch.compile 配合 JIT 优化经常能带来拔群的效果，以至于常常听到“手写算子已经没必要了”的论调。

本文直接聚焦一个核心命题：为什么“手写算子（hand-written operator）”与“内核融合（kernel fusion）”能够带来大幅度的性能提升？本文将基于大模型标配的 RoPE (Rotary Position Embedding) 算子，对比 PyTorch 朴素实现、PyTorch 查表缓存实现，以及单 CUDA Kernel 的手写实现，用底层逻辑和测试数据给出答案。

完整代码可参见链接：

简短结论：

- 单纯在 PyTorch 层做表缓存（cos/sin）可以减掉一部分开销,无法解决根本的访存瓶颈。
- 手写内核有核心的、不可替代的优势
  - 打破读写放大：极致减少对输入特征 `q` 的读取次数（大模型推理的终极瓶颈永远在显存带宽）。
  - 以算代读：降低空间复杂度，通过寄存器计算换取宝贵的显存带宽。
  - 将读、算、写操作彻底融合在单一 Kernel 内部，完全消除中间变量的显存往返（同时省去 2~5 ns 的 launch 开销，虽在 CUDA Graph 时代这已不是主要痛点，但显存复用的收益依然巨大）。

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

在 pytorch fp32 实现中，即使做了 COS/SIN 表 cache，由于框架底层 Operator 调用的物理隔离，依然存在灾难性的读写放大：
在执行 q_embed = (q *cos) + (rotate_half(q)* sin) 时，输入 $q$ 的前后半段需要分别被提取、拼接并与另一个 tensor 相乘。

粗略估算一下，为了完成这一步，显存读写量包括：

- $q$ 的多次读取。
- SIN/COS 表的读取（通常是 $q$ 的两倍量）。
- 临时空间分配（如 rotate_half(q)、q *cos、...* sin 产生的大量中间 tensor）带来的写出和再读取。
保守估计，总数据搬运量高达 $bs \times seq\_len \times head\_dim \times 4 \text{ bytes} \times 11$ 左右（当然，这是我最原始的粗略估计，pytorch可能默认有优化策略数据量和估计有出入）。由于框架层面的限制，带宽被无意义的中间变量榨干了。

而手写算子的降维打击在于：

- One Pass：只读一遍 $q$，kernel内部实时计算 sin/cos（以算代读），也绝不浪费显存带宽去读取预存的三角表和中间变量。
- 理论读写总量被硬核压缩到了仅仅 $bs \times seq\_len \times head\_dim \times 4 \text{ bytes} \times 2$（一读一写）。直接削减了约 80%+ 的无效访存！

## 1. 手写算子的工程实践

针对 Memory-bound 算子，我们出两招：

- 向量化 Load/Store (128-bit / float4)
  - 无论是 CPU 还是 GPU，访存速度永远是被计算单元按在地上摩擦的。由于目前 LLM 的 head_size 必然是 4 的倍数，我们可以直接使用 float4 进行合并内存访问。
  - 一次性吃进 4 个 float：4 次总线事务 → 1 次总线事务。在寄存器里直接完成 4 个元素的旋转，然后一把写回。

- 内核融合（Kernel Fusion）
  - 强行把 Load、Compute (__sincosf)、Store 揉进一个 Kernel，避免中间结果落地显存。

结合这两点，核心 C++ 代码如下：

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

测试中保留了 rope_with_sin_cos_cache 逻辑，但这只是对框架原生开销的修补。PyTorch 层的 Cache 无法改变底层的内存访问粒度和线程对齐。缓存只是对 Baseline 的略微增强，而非手写内核的平替。

## 2. Benchmark 测试与深度剖析

测试设备为 RTX 5060 移动版。我们对是否开启 torch.compile 做了严谨对比。

- Baseline (不开启 torch.compile):

```yaml
torch                           mean time: 18.794270 ms  | 有效带宽:  57.13 GB/s
torch.rope_with_sin_cos_cache   mean time: 17.558725 ms  | 有效带宽:  61.15 GB/s
rope                            mean time:  3.430634 ms  | 有效带宽: 312.98 GB/s
rope_fp32x4 (手写向量化)         mean time:  3.383279 ms  | 有效带宽: 317.36 GB/s
```

- JIT 优化对比 (开启 torch.compile):

```yaml
torch                           mean time:  5.242156 ms  | 有效带宽: 204.83 GB/s
torch.rope_with_sin_cos_cache   mean time:  4.811796 ms  | 有效带宽: 223.15 GB/s
rope                            mean time:  3.331872 ms  | 有效带宽: 322.26 GB/s
rope_fp32x4 (手写向量化)         mean time:  3.306061 ms  | 有效带宽: 324.78 GB/s
```

### 2.1 带宽利用率与极致榨取：接近极限的硬核答卷

为了客观评估算子的真实表现，我们引入了有效带宽（Effective Bandwidth）的计算。手写算子的实际吞吐为 128*8192*128*2*4/3.306061*1e3/1e9 ~= 325 GB/s。

作为参考，我使用 nvbandwidth 测试了显卡底层的双向纯拷贝，实测物理极限约为 337 GB/s。这意味着，在端到端的真实调用环境下，我们的 Kernel 实际有效带宽利用率达到了 325 / 337 = 96.4%！

有趣的是，在不开启 torch.compile 的 Baseline 测试中，PyTorch 查表缓存版（cache）相比朴素版的性能提升非常有限（有效带宽仅从 57.13 GB/s 升至 61.15 GB/s）。这进一步印证了我们的推论：在这个数据规模下，RoPE 的核心瓶颈根本不在三角函数计算，而在于极其严重的显存带宽拥堵。查表不仅没有缓解拥堵，反而可能因为多读了一张表加剧了访存负担。

当然，受限于移动端显卡的功耗墙、动态频率以及虚拟化环境，绝对时间（ms）会有较大抖动，因此我们更应该关注不受频率影响的 NCU 硬件级指标。NCU profile 也显示手写kernel compute memory 吞吐达到了97%，说明向量化读取和“以算代读”的策略非常有效，L1 层面每一份搬运的数据都得到了接近极限的计算利用。

因此，目前的框架自动优化措施（如 torch.compile）固然强大，但想把硬件最后一点性能榨干的场景时，终究还是无法达到手写 Kernel 的极致水平。这，就是手写算子在今天的核心意义所在。

## 3. 总结与讨论

我们手写的 rope kernel 通过向量化读取、以算代读等手段最大化提高带宽利用效率，性能领先于 torch.compile 优化的 naive 版本。而且本文仅仅是以 rope 为例说明手写算子的一些优势，并未对 rope 的实现做更深入的优化和分析，cos/sin 可以提前算好通过常量/纹理内存进行加速，以算代读中寄存器计算的结果可以循环多次复用等等，实际上 rope 还常常和线性层或和 attention 融合计算，不过就不多讨论了。此外对于一些自定义的 op，或者复杂度更高的 op，目前的自动优化措施终究无法达到手写 kernel 的极致水平。

如有错误，欢迎指正！

以上，共勉。
