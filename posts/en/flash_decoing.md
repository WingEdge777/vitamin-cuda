# [CUDA in Practice] Hand-Rolled Flash Decoding on SM120: Beating `flashinfer.single_decode_with_kv_cache`
>
> Note: Text translated by AI. Code crafted by human.
>
> This article is intended for readers with a solid CUDA foundation, familiar with GEMM/multi-head-attention optimization, and interested in advanced inline PTX tuning.
>
> Full kernel and test code: [flash_decode](/kernels/flash_decode)

## 0. Preface — Decode vs. Prefill Attention: Two Different Optimization Philosophies

Following the previous FMHA post: that one focused on prefill FlashAttention; this one switches to decode attention (single query + long KV cache) and discusses how to write a practical kernel for that regime.

Baselines used in this article:

- PyTorch native implementation after `torch.compile` (general-purpose baseline)
- flashinfer's `single_decode_with_kv_cache` (existing specialized baseline)

A note on the flashinfer baseline: I added a minimal compatibility fix so it can run normally on my RTX 5060 Laptop (26 SMs). Details are below.

### PyTorch Native Baseline

```python
@torch.compile
def torch_native_decode(q, k, v, scale=None):
    # q: [head, dim] -> [32, 128]
    # k: [seq, head, dim] -> [4096, 32, 128]
    # v: [seq, head, dim] -> [4096, 32, 128]
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    # reshape for batched GEMV-style compute
    q_b = q.unsqueeze(1)      # [32, 1, 128]
    k_b = k.permute(1, 2, 0)  # [32, 128, 4096]
    v_b = v.transpose(0, 1)   # [32, 4096, 128]

    attn_scores = torch.matmul(q_b, k_b) * scale
    attn_probs = torch.softmax(attn_scores, dim=-1)
    out = torch.matmul(attn_probs, v_b)
    return out.squeeze(1)
```

Don't underestimate this baseline just because it's "PyTorch code."
Decode attention is fundamentally close to batched GEMV, where mature library paths plus `torch.compile` graph optimization can be very strong.

### A Minimal Compatibility Fix for flashinfer

I found `cudaOccupancyMaxActiveBlocksPerMultiprocessor` unexpectedly returning `0` in flashinfer's `SingleDecodeWithKVCacheDispatched` path on my setup.

My temporary fix:

- force `num_blocks_per_sm = 1` (its default double K/V buffering already uses about 64KB SMEM anyway)
- set `max_num_kv_chunks = 1` to avoid downstream zero-chunk behavior

Also, once I saw flashinfer's double K/V-buffer SMEM footprint, I already expected occupancy loss to hurt it on this card.

### Kernel Outline in This Post

- `flash_decode_tma_128`
  - `BN=64`, TMA + `float4` vectorized SMEM loads + online softmax
- `flash_decode_tma_dbf_k`
  - same as above, plus double K buffers

Both target MHA decode attention with `head_dim = 128`.

## 1. Flash Decoding Recap

At decoding stage, `batch` and `q_seq_len` are tiny (often effectively 1), so sequence parallelism on Q side vanishes and SM utilization drops.

Classic flash-decoding idea:

- split KV sequence into chunks
- compute chunk-local online stats (`m_i`, `d_i`, partial `o`)
- merge chunk results in a second pass

For clarity, this post discusses one decode step with:

- `q` shape `[head, dim]`
- `kv` shape `[seq, head, dim]`

### Tiling Strategy

- Keep `BN=64` tile size from prior FlashAttention experiments
- On my card (100KB total SMEM, 48KB per block), this is practical and stable

So each KV loop iteration loads `64 x 128` K/V tiles.

### Block/Grid Strategy

- block size = 128
  - 32/64 underutilize warp-level flexibility
  - 256 is unnecessary for this kernel's compute density
  - 128 tested best
- grid:
  - `y`: head
  - `x`: chunked KV sequence

Chunk size heuristic:

- target about `2 * num_sms` total blocks for this card (`26 * 2 = 52`)
- derive runtime `chunk_size` from `head * seq / 52`
- round up to power-of-two-style buckets (`256/512/1024/2048`)

### Why No Tensor Core Path Here?

Decode's `q` is effectively one row.
`mma m16n8k16` requires `m=16`; padding fake rows wastes work.
This problem is bandwidth-dominated, so efficient K/V movement and consumption matter more than forcing MMA.

So I skip `mma`/`ldmatrix` path and instead:

- use TMA to move full `64 x 128` K/V tiles to SMEM
- use vectorized reads (`float4`) to maximize bandwidth usage

## 2. Kernel Details

Within one tile/chunk:

- load `Q[1,128]`
- compute `S = Q * K^T` (`1 x 64`)
- apply softmax to get `P[1,64]`
- compute `O = P * V`

How to split this over 128 threads (4 warps)?

Bad idea first (rejected): one thread computes an entire row/column dot product.
That violates CUDA parallel-first design and explodes both sync and bank-conflict pressure.

Chosen strategy:

- one warp handles row groups
- with half precision and `float4` vectorization, each thread handles 8 elements
- one warp can process two rows per step; loop depth is reduced

### Pass Structure

Pass 1 (chunk kernel):

- init K/V SMEM buffers + TMA mbarriers
- load `Q` fragment into registers
- maintain subgroup-local online softmax state (`acc_o`, `m_i`, `d_i`)
- loop through tiles inside chunk:
  - TMA load K/V
  - compute local attention scores
  - compute weighted partial outputs
  - online merge into subgroup state
- block-level merge subgroup states
- write split results:
  - `ws_o` (partial output)
  - `ws_lse` (`logsumexp` form)

Pass 2 (reduce kernel):

- merge all chunk outputs using `ws_lse` weights
- produce final output `o`

### Using `lse` Instead of Storing `m_i` + `d_i` Separately

I use:

```c++
lse = m_i * ln(2) + ln(d_i)
```

Then in pass 2:

```c++
float max_lse = max(lse_i);
float global_lse = max_lse + ln(sum(exp(lse_i - max_lse)));
o = sum(ws_o_i * exp(lse_i - global_lse));
```

This makes cross-chunk merge cleaner.

### Notes

- No `lazy_rescale` here:
  - in prefill it replaced expensive repeated scaling on larger accumulators
  - here `acc_o` is small and trade-off is not favorable
- No `need_causal_mask`:
  - in decode, all current KV positions are visible to current query

### Grouped Warp/Block Reduce

Each warp effectively behaves like two 16-thread groups for reduction, so reduction helpers are parameterized with group width 16.

This implementation is still rough around the edges (e.g., `ws_o/ws_lse` write/read patterns are not deeply optimized yet), but the main path is working and fast.

## 3. `flash_decode_tma_dbf_k`: Double-K Buffering

After finishing baseline `flash_decode_tma_128`, I optimized further.

Current resource picture:

- ~32KB SMEM + barriers + temporary reduction storage
- occupancy can still keep 2 resident blocks

Given that, the most practical next step is additional pipelining to hide latency.

### 3.1 Double K Buffer

Use double buffering for K only, keep V single-buffered.

Why this works:

- K and V are already consumed at different moments in the pipeline
- V latency is partially hidden by `QK`/softmax work
- adding another K buffer improves overlap on the K side too

Pipeline outline:

- initialize 2 K tiles
- prologue: preload `Ks[0]`
- loop:
  - prefetch next K tile if available
  - issue V transfer
  - wait current K tile, run online softmax path
  - wait V, compute output update
  - sync and flip read/write indices

### 3.2 Epilogue Tweak

Original epilogue had repeated block-reduce loops.
I switched to reusing K/V SMEM buffers as staging storage, then a dedicated 16-thread group performs final accumulation/writeback.

Gain is modest, but change is kept for cleaner staging behavior.

## 4. Benchmark

The `flash-infer` numbers below come from the locally patched version described above, not an untouched upstream build.

```yaml
####################################################################################################
decode, kv seq: 8192, head: 32, dim: 128
torch.compile                            mean time: 0.454655 ms, 295.24 GB/s
flash-infer                              mean time: 0.403291 ms, speedup: 1.13, GB/s: 332.85
flash_decode_tma_128                     mean time: 0.408378 ms, speedup: 1.11, GB/s: 328.70
flash_decode_tma_dbf_k_128               mean time: 0.366698 ms, speedup: 1.24, GB/s: 366.06
####################################################################################################
decode, kv seq: 16384, head: 32, dim: 128
torch.compile                            mean time: 0.872882 ms, 307.55 GB/s
flash-infer                              mean time: 0.784423 ms, speedup: 1.11, GB/s: 342.23
flash_decode_tma_128                     mean time: 0.735274 ms, speedup: 1.19, GB/s: 365.10
flash_decode_tma_dbf_k_128               mean time: 0.733273 ms, speedup: 1.19, GB/s: 366.10
####################################################################################################
decode, kv seq: 32768, head: 32, dim: 128
torch.compile                            mean time: 1.507921 ms, 356.04 GB/s
flash-infer                              mean time: 1.499479 ms, speedup: 1.01, GB/s: 358.05
flash_decode_tma_128                     mean time: 1.495797 ms, speedup: 1.01, GB/s: 358.93
flash_decode_tma_dbf_k_128               mean time: 1.455790 ms, speedup: 1.04, GB/s: 368.79
####################################################################################################
decode, kv seq: 65536, head: 32, dim: 128
torch.compile                            mean time: 2.980080 ms, 360.31 GB/s
flash-infer                              mean time: 2.897006 ms, speedup: 1.03, GB/s: 370.64
flash_decode_tma_128                     mean time: 2.856871 ms, speedup: 1.04, GB/s: 375.85
flash_decode_tma_dbf_k_128               mean time: 2.849400 ms, speedup: 1.05, GB/s: 376.84
####################################################################################################
decode, kv seq: 131072, head: 32, dim: 128
torch.compile                            mean time: 6.044398 ms, 355.29 GB/s
flash-infer                              mean time: 5.751600 ms, speedup: 1.05, GB/s: 373.37
flash_decode_tma_128                     mean time: 5.663495 ms, speedup: 1.07, GB/s: 379.18
flash_decode_tma_dbf_k_128               mean time: 5.736955 ms, speedup: 1.05, GB/s: 374.33
####################################################################################################
decode, kv seq: 131073, head: 32, dim: 128
torch.compile                            mean time: 6.466227 ms, 332.11 GB/s
flash-infer                              mean time: 6.117131 ms, speedup: 1.06, GB/s: 351.07
flash_decode_tma_128                     mean time: 5.701174 ms, speedup: 1.13, GB/s: 376.68
flash_decode_tma_dbf_k_128               mean time: 5.695415 ms, speedup: 1.14, GB/s: 377.06
```

Takeaways:

- both custom kernels consistently beat `torch.compile` baseline
- `flash_decode_tma_dbf_k_128` is best overall
- double-K buffering gives bigger gains on shorter sequences
- for longer sequences, all kernels approach bandwidth ceiling and gaps naturally shrink
- peak logical bandwidth reaches `377.06 / 384 = 98.2%`, very close to this GPU's theoretical peak

In this patched setup on my GPU, flashinfer effectively runs at single-block occupancy with double K/V buffers, which is weaker than my `2 blocks + 2K + 1V` configuration here. This again shows how critical occupancy is when it gets too low.

NCU screenshots:

- `https://cdn.jsdelivr.net/gh/WingEdge777/CDN@main/images/vitamin_cuda/flash_decoding_summary.png`
- `https://cdn.jsdelivr.net/gh/WingEdge777/CDN@main/images/vitamin_cuda/flash_decoding_detail.png`

There are still some uncoalesced global accesses, mainly from `ws_o/ws_lse` I/O. They are outside the hot loop, and DRAM bandwidth is already 90%+, so total runtime impact is limited.

## 5. End

That's my current understanding and implementation of flash decoding.

There are still rough edges, but I'll leave them for now and move on to other experiments.

If you spot mistakes, feel free to correct me. Suggestions are also welcome.

That's all.
