# [CUDA in Practice] RoPE — Why Kernel Fusion in Hand-Written Operators Matters: Reducing Memory Traffic and Launch Overhead
>
>Note: Text translated by AI. Code crafted by human.
>

AI compilers are evolving fast. In many cases, `torch.compile` plus JIT optimization in PyTorch can deliver striking speedups, to the point that people often say, "hand-written operators are no longer necessary."

This article focuses on one core question: why can **hand-written operators** and **kernel fusion** still deliver major performance gains? Using the RoPE (Rotary Position Embedding) operator that appears almost everywhere in large models, we compare a naive PyTorch implementation, a PyTorch version with cached cosine/sine tables, and a hand-written single-kernel CUDA implementation. The goal is to answer that question with low-level reasoning and benchmark data.

The complete kernel and test code can be found at [rope](https://github.com/WingEdge777/vitamin-cuda/tree/main/kernels/rope).

Short conclusion:

- Caching `cos`/`sin` tables at the PyTorch level removes some overhead, but it does not solve the fundamental memory traffic bottleneck.
- Hand-written kernels still have core advantages that are hard to replace:
  - They break the read/write amplification pattern and drastically reduce how many times the input feature `q` must be loaded. In large-model inference, the ultimate bottleneck is still memory bandwidth.
  - They trade compute for memory loads, reducing space complexity and spending registers to save precious DRAM bandwidth.
  - They fuse load, compute, and store inside a single kernel, eliminating intermediate round-trips to global memory entirely. This also removes 2-5 ns of launch overhead. In the CUDA Graph era that is no longer the main pain point, but the memory reuse benefit is still significant.

## 0. Analyzing the Naive PyTorch Implementation

Code first:

```python

def compute_default_rope_parameters(head_dim):
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2).float().cuda() / head_dim)
    )  # 64
    return inv_freq

# Precompute freqs
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

# NeoX-style RoPE, assume a single head
# multi-head can be folded into the batch dimension
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

Even in the FP32 PyTorch implementation, and even with cached `COS`/`SIN` tables, the physical separation between framework-level operators still causes disastrous read/write amplification. During

`q_embed = (q * cos) + (rotate_half(q) * sin)`

the first and second halves of `q` must be extracted separately, concatenated, and multiplied with another tensor.

A rough estimate of the global-memory traffic needed for this step includes:

- Multiple reads of `q`
- Reads of the `SIN`/`COS` tables, often about 2x the size of `q`
- Temporary allocations such as `rotate_half(q)`, `q * cos`, and `... * sin`, each of which causes extra writes and later re-reads

Even conservatively, the total data movement can reach roughly:

`bs * seq_len * head_dim * 4 bytes * 11`

Of course, this is only my original back-of-the-envelope estimate. PyTorch may apply internal optimizations, so the real number may differ. But the point stands: at the framework level, bandwidth gets consumed by meaningless intermediate tensors.

This is where the hand-written kernel lands a direct hit:

- One Pass: read `q` only once, compute `sin`/`cos` on the fly inside the kernel, and never waste bandwidth reading precomputed trig tables or intermediate tensors.
- The theoretical total read/write volume is compressed down to just:

`bs * seq_len * head_dim * 4 bytes * 2`

That is one read and one write only, cutting roughly 80%+ of the wasted memory traffic.

## 1. Practical Hand-Written Kernel Engineering

For a memory-bound operator, we use two main tricks:

- Vectorized Load/Store (`128-bit` / `float4`)
  - On both CPUs and GPUs, memory access is always far slower than compute units. Since `head_size` in current LLMs is naturally a multiple of 4, we can directly use `float4` for packed memory access.
  - Load 4 floats at once: 4 bus transactions become 1. Perform the rotation directly in registers, then write the result back in one shot.

- Kernel Fusion
  - Force `Load`, `Compute` (`__sincosf`), and `Store` into one kernel so no intermediate result ever lands in global memory.

Combining the two gives the following core C++ code:

```c++
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])
// RoPE NeoX float4 version
// a[bs, seq_len, head_dim]
__global__ void rope_fp32x4_kernel(float *a, float *b, int seq_len, int head_dim) {
    int pos = blockIdx.x * 4;
    int tid = threadIdx.x;
    pos += tid * 4 / (head_dim >> 1);
    int f_idx = tid * 4 % (head_dim >> 1);
    int idx = blockIdx.y * (seq_len * head_dim) + pos * head_dim + f_idx;
    // Packed load
    float4 x = LDST128BITS(a[idx]);
    float4 y = LDST128BITS(a[idx + (head_dim >> 1)]);

    float inv_freq[4], c[4], s[4];
    // Compute the rotation angles
#pragma unroll
    for (int i = 0; i < 4; i++) {
        inv_freq[i] = 1.f / __powf(theta, (2.0f * (f_idx + i)) / head_dim);
        __sincosf(pos * inv_freq[i], &s[i], &c[i]);
    }
    // Rotate
    float4 x_new, y_new;
    x_new.x = x.x * c[0] - y.x * s[0];
    x_new.y = x.y * c[1] - y.y * s[1];
    x_new.z = x.z * c[2] - y.z * s[2];
    x_new.w = x.w * c[3] - y.w * s[3];
    y_new.x = y.x * c[0] + x.x * s[0];
    y_new.y = y.y * c[1] + x.y * s[1];
    y_new.z = y.z * c[2] + x.z * s[2];
    y_new.w = y.w * c[3] + x.w * s[3];

    // Packed store
    LDST128BITS(b[idx]) = x_new;
    LDST128BITS(b[idx + head_dim / 2]) = y_new;
}
```

### 1.1 A Note on `cos`/`sin` Caching

The benchmark still keeps the `rope_with_sin_cos_cache` path, but that only patches over part of the framework overhead. A PyTorch-level cache cannot change the underlying memory access granularity or thread alignment. Caching is only a small enhancement to the baseline, not a substitute for a hand-written kernel.

## 2. Benchmark Results and Deeper Analysis

The test device is an RTX 5060 Mobile GPU. We compare both with and without `torch.compile`.

- Baseline (without `torch.compile`):

```yaml
torch                           mean time: 18.794270 ms  | effective bandwidth:  57.13 GB/s
torch.rope_with_sin_cos_cache   mean time: 17.558725 ms  | effective bandwidth:  61.15 GB/s
rope                            mean time:  3.430634 ms  | effective bandwidth: 312.98 GB/s
rope_fp32x4 (hand-written vectorized)   mean time:  3.383279 ms  | effective bandwidth: 317.36 GB/s
```

- JIT comparison (with `torch.compile`):

```yaml
torch                           mean time:  5.242156 ms  | effective bandwidth: 204.83 GB/s
torch.rope_with_sin_cos_cache   mean time:  4.811796 ms  | effective bandwidth: 223.15 GB/s
rope                            mean time:  3.331872 ms  | effective bandwidth: 322.26 GB/s
rope_fp32x4 (hand-written vectorized)   mean time:  3.306061 ms  | effective bandwidth: 324.78 GB/s
```

### 2.1 Bandwidth Utilization: Squeezing the Hardware to the Limit

To evaluate the operator objectively, we use **effective bandwidth**. The actual throughput of the hand-written kernel is:

`128 * 8192 * 128 * 2 * 4 / 3.306061 * 1e3 / 1e9 ~= 325 GB/s`

For reference, I used `nvbandwidth` to measure the GPU's low-level bidirectional copy throughput, and the physical limit came out to about `337 GB/s`. That means in a real end-to-end runtime environment, our kernel reaches:

`325 / 337 = 96.4%`

effective bandwidth utilization.

Another interesting point: in the baseline test without `torch.compile`, the PyTorch version with trig-table cache improves only slightly over the naive version, from `57.13 GB/s` to `61.15 GB/s`. That further supports the core argument here: at this problem size, the bottleneck in RoPE is not trig computation at all, but severe DRAM bandwidth congestion. The lookup table does not relieve that congestion. It may even make memory traffic worse by adding another table read.

Of course, on a mobile GPU, the absolute runtime in milliseconds can fluctuate significantly because of power limits, dynamic clocks, and virtualization overhead. So instead of staring only at wall-clock time, we should pay more attention to hardware-level NCU metrics that are less sensitive to frequency variation. NCU profiling also shows the hand-written kernel reaching about `97%` compute-memory throughput, which indicates that vectorized loads and the "compute instead of read" strategy are both highly effective. At the L1 level, almost every byte moved is being turned into useful work.

So yes, modern framework-side optimization like `torch.compile` is powerful. But when the goal is to squeeze out the last bit of hardware performance, it still cannot reach the same extreme level as a hand-written kernel. That is the core reason hand-written operators still matter today.

## 3. Summary and Discussion

The RoPE kernel in this article improves bandwidth efficiency through vectorized memory access and by trading compute for memory reads, and it clearly outperforms the `torch.compile`-optimized naive version. Also, this article uses RoPE only as an example to illustrate some of the advantages of hand-written operators. It is not an attempt to fully optimize RoPE itself. There is still plenty of room left: `cos`/`sin` could be precomputed and accelerated through constant memory or texture memory; intermediate results produced by the compute-for-read strategy could be reused multiple times in registers; and in practice, RoPE is often fused further with linear layers or with attention itself.

More broadly, for custom operators or operators with higher algorithmic complexity, current automatic optimization still cannot reach the extreme ceiling of a carefully hand-written kernel.

If there are any errors, please feel free to correct them. The complete kernel and test code can be found at [rope](https://github.com/WingEdge777/vitamin-cuda/tree/main/kernels/rope).
