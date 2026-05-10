# [CUDA in Practice] FMHA on SM120: Beating `torch.sdpa` (FlashAttention-2)
>
> Note: Text translated by AI. Code crafted by human.
>
> This article is intended for readers with a solid CUDA foundation, familiar with GEMM/multi-head-attention optimization, and interested in advanced Tensor Core / inline PTX tuning.
>
> It's best to read my previous posts on HGEMM SM120, safe online softmax, and the GEMM series first. Some concepts are reused and only briefly mentioned here.
>
> Full kernel and test code: [flash_attn](/kernels/flash_attn)

## 0. Preface — The Crown Jewel Operator in the LLM Era

In the LLM era, almost every model relies on Multi-Head Attention. Its complexity scales quadratically with sequence length, so attention is often the dominant compute cost.

Even though many linear-attention variants have appeared, they still usually cannot match full attention accuracy. So in practice, people move toward hybrid attention layers. That topic is out of scope here.

This post has one straightforward goal: on an RTX 5060 Laptop (SM120), hand-roll a high-performance FMHA kernel without third-party libraries (no CUTLASS, no CUTE), using just TMA + `ldmatrix` + `mma`.

And yes, it beats FA2 in my setup — not because I'm stronger than Tri Dao, but because FA2's kernel targets SM80 and does not have SM120's TMA advantages. If Tri Dao wrote this exact TMA+`ldmatrix`+`mma` version, I'd probably lose cleanly.

The overall implementation idea is still close to FlashAttention-2 (I did not deeply inspect the FA2 code itself; the paper is enough for this post). We won't re-derive paper-level theory here.

Kernel scope in this article:

- `fmha_tma_128` (`BM x BN = 64 x 64`, TMA + `ldmatrix` + `mma`)

That's it. One kernel. Causal attention with `head_dim = 128` (a very common setting).

I initially considered introducing a progressive path (three-matrix multiply first, then online softmax), but even that became large enough to be noisy. So I present the final kernel directly.

## 1. FlashAttention Recap

FlashAttention is one of my favorite engineering works:

- FA1: online softmax in shared memory
- FA2: improved parallelization and access pattern
  - especially by shifting part of the inner-loop reduction from V to Q, plus better sequence parallelism

Very solid work.

For MHA:

`MHA = softmax(q @ k.T * scale) @ v`, where `scale` is usually `1/sqrt(head_dim)`.

Naive implementation:

- `s = q @ k.T * scale`
- `p = softmax(s)`
- `o = p @ v`

This introduces repeated global memory traffic (`s` write/read, `p` read, etc.).

FlashAttention eliminates intermediate `s/p` global-memory traffic by doing tiled online softmax in shared memory and writing final `o` directly.

So ideally:

- one global read for `q/k/v`
- one global write for `o`

Assume `q/k/v` shapes are `[batch_size, seq_len, heads, head_dim]` (my kernel also supports GQA), and output `o` has the same shape as `q`.

High-level loop:

- load one `q` tile
- loop over sequence dimension, loading one `kv` tile each iteration
- compute `q @ k.T`, then online-softmax updates, then `p @ v`, accumulate into `o`
- write final `o` tile back to global memory

### Tiling Strategy

Practical constraints first:

- `head_dim=128` is loaded as a whole (no split on K dimension for score dot-product semantics)
- sequence length must be tiled
- `heads` and `batch` go naturally to grid dimensions

Given SMEM limits on my card (total 100KB, 48KB per block), after repeated tuning:

- `BM = 64` for `q` sequence tile
- `BN = 64` for `kv` sequence tile

Check:

- `(BM*head_dim + BN*head_dim*2) * 2 bytes`
- `= (64*128 + 64*128*2) * 2 = 48KB` (right at block limit, plus a few mbarriers)

This allows two active blocks, keeping occupancy reasonable.

Thread-block setup:

- block size = 128
- GEMM shape in inner core: `64 x 64 x 128`
- `mma` instruction: `m16n8k16`
- 4 warps are enough

Grid mapping:

- `x`: q-sequence tiles (for better L2 reuse over KV loop)
- `y`: batch
- `z`: head

Rationale: neighboring blocks should read overlapping KV tiles whenever possible.

## 2. FMHA Kernel Implementation

With data/thread tiling fixed, kernel flow is:

- initialize shared-memory buffers `Qs/Ks/Vs` and 3 TMA mbarriers
- initialize output accumulator registers `acc_o`
- load `Qs` and wait
- loop over KV tiles:
  - issue TMA for `Ks` and `Vs`
  - wait for `Ks`, immediately compute `QK` (overlapping with `Vs` transfer)
  - run online softmax, producing `P`
  - register-layout transform for `P` so it can feed `P @ V` MMA directly
  - compute and accumulate `O`
- final writeback via shared-memory staging

Several notable tricks:

### 2.1 Lazy Rescale

In strict online softmax, `acc_o` would be rescaled every iteration.
I delay that using row-wise scale registers and only rescale when threshold is hit.

I use threshold `2^-8`:

- final `P` is converted to BF16 for Tensor Cores
- BF16 has 7-bit mantissa
- below this threshold, reciprocal scaling amplifies quantization error too aggressively

So `2^-8` is the practical trade-off between branch overhead and BF16 precision.

### 2.2 `k_end` and Causal Mask

In prefill causal attention:

- strictly upper triangle is always zero and should be skipped
- fully valid left area participates normally
- only diagonal-region tiles need masking checks

### 2.3 Warp-Level Reduce for `m_i / d_i`

Because of `m16n8k16` fragment mapping, 32 threads are naturally partitioned into 8 groups of 4.
Two `__shfl_xor_sync` steps (masks 1 and 2) are enough for row-local reduction in each 4-thread group.

### 2.4 Register Reuse: `Ps` Reordering

My first version wrote `P` to shared memory, then read it back with `ldmatrix.x4`.
After validating fragment layouts, I found direct register reuse is possible:

- `QK`'s fragment-C register layout aligns with what `P@V` needs for fragment-A (under the outer-tile iteration structure)

So I directly pack adjacent `float` pairs with `__float22bfloat162_rn` and skip the extra smem roundtrip.

### 2.5 Overlap `Vs` Transfer with `QK` Compute

Initially I loaded `Ks/Vs` together and waited too conservatively.
Then I realized `QK + softmax` does not depend on `Vs`.

So:

- add a dedicated `mbarrier` for `Vs`
- overlap `Vs` transfer with `QK` compute

This turned out to be a key reason this kernel surpasses FA2 in my tests.

## 3. Benchmark

Results against `torch.sdpa` (FA2 backend) in prefill bf16 mode with `head=32`, `dim=128`:

```yaml
####################################################################################################
prefill, batch:  1, seq: 512, head: 32, dim: 128
torch.sdpa (FA2 backend)                 mean time: 0.121095 ms, 17.73 tflops
fmha_tma_128                             mean time: 0.098975 ms, speedup: 1.22, tflops: 21.70
####################################################################################################
prefill, batch:  1, seq: 1024, head: 32, dim: 128
torch.sdpa (FA2 backend)                 mean time: 0.399812 ms, 21.48 tflops
fmha_tma_128                             mean time: 0.352873 ms, speedup: 1.13, tflops: 24.34
####################################################################################################
prefill, batch:  1, seq: 2048, head: 32, dim: 128
torch.sdpa (FA2 backend)                 mean time: 1.355930 ms, 25.34 tflops
fmha_tma_128                             mean time: 1.293091 ms, speedup: 1.05, tflops: 26.57
####################################################################################################
prefill, batch:  1, seq: 4096, head: 32, dim: 128
torch.sdpa (FA2 backend)                 mean time: 4.983902 ms, 27.58 tflops
fmha_tma_128                             mean time: 4.891181 ms, speedup: 1.02, tflops: 28.10
####################################################################################################
prefill, batch:  1, seq: 8192, head: 32, dim: 128
torch.sdpa (FA2 backend)                 mean time: 18.248972 ms, 30.13 tflops
fmha_tma_128                             mean time: 17.612578 ms, speedup: 1.04, tflops: 31.21
```

Across all tested sequence lengths, this kernel beats FA2 in my environment.
The main reasons:

- TMA reduces register pressure (especially addressing overhead)
- better overlap between copy and compute

NCU report screenshots:

- `summary`: `https://cdn.jsdelivr.net/gh/WingEdge777/CDN@main/images/vitamin_cuda/fmha_summary.png`
- `shared memory`: `https://cdn.jsdelivr.net/gh/WingEdge777/CDN@main/images/vitamin_cuda/fmha_shared_table.png`

Highlights:

- still has register headroom (FA2 tends to be register-maxed)
- L2 utilization reaches 90%+

### 3.1 Discussion

- There are still non-fused FP32 ops (mostly around softmax flow)
  - I reduced some with tricks (`exp2(fmaf)`, lazy rescale), but cannot eliminate all standalone add/mul meaningfully.
- There is still a small amount of bank conflict (not the classic static swizzle-address conflict pattern)
  - likely tied to post-unroll dynamic instruction ordering
  - behavior shifts when surrounding code changes (e.g., adding lazy rescale)
- FA's NCU profile still looks cleaner:
  - no non-fused FP32 ops
  - essentially zero bank conflict
  - likely aided by CUTLASS/SASS-level optimizations and a different tiling strategy

Current kernel flow is less straightforward due to many practical tricks. I may refactor for readability later.

## TODO

Attention has a lot of room left:

- optimize causal-mask path (current `need_causal_mask` logic expands unrolled instructions)
- decode attention
  - prefill is compute-heavy; decode is bandwidth-bound (GEMV style), so optimization philosophy changes completely
- attention with KV-cache blocks (paged attention core requirement for integration with real inference systems)
- FP8 mixed precision + KV quantization
  - leverage SM120 FP8 and explore INT8/INT4 KV cache compression

I may move to decode attention next, or take a break first.

## End

That's the full implementation journey for my SM120 FMHA kernel.

Complete code and tests are available in the GitHub repo.

That's all.
