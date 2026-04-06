# Numbers Every CUDA Developer Should Know
>
>Note: Text translated by AI. Code crafted by human.
>
This post is a cheat sheet for CUDA programmers: the hardware constants and latency scales that really matter when you care about performance.

## 0. Preface

In HPC and deep learning, writing CUDA that “works” is easy; writing CUDA that is genuinely fast demands a solid mental model of the hardware.

Just as Jeff Dean’s famous “numbers every programmer should know” anchor intuition about systems, GPU programming has its own golden figures. Ignore them and you may leave most of the chip idle; internalize them and you stand a much better chance of using the hardware as it was designed to be used.

Below is a concise tour of the constants and order-of-magnitude facts that shape CUDA performance.

## 1. The execution atom: warp size = 32

This is the defining number in CUDA. A warp is the smallest scheduling unit on the GPU.

- Meaning: an SM issues one instruction to 32 threads at a time that execute the same opcode (SIMT: single instruction, multiple threads).
- Performance implications:
  - Branch divergence: if the 32 threads in a warp diverge at the same instruction, the hardware still runs in SIMT fashion: different paths are walked under predicates/masks, the active set changes as paths are taken, and throughput is discounted versus a fully active warp. Volta and later independent thread scheduling mainly refines *within-warp* synchronization and reconvergence semantics and relaxes some constraints around primitives like `__syncwarp()`, but a warp is still one instruction broadcast to 32 lanes—divergence still costs; do not read “independent scheduling” as “free divergence.”
  - Memory coalescing: you only get best memory throughput when threads in a warp hit contiguous, suitably aligned addresses.
  - Active masks: warp-wide intrinsics (e.g. `__shfl_sync`) use 32-bit lane masks.
  - Tail warps: if your total thread count is not a multiple of 32, the last warp has inactive lanes; those lanes still ride along with the warp on the hardware.

The practical read: if a launch has very few *active* threads, you not only underfill the SM—you also get tail warps where some lanes do no useful work but still consume scheduling slots, which makes memory and instruction latency harder to hide. In plain terms: aim for enough warps in flight and shape the problem so you fully utilize the SIMD width as often as possible. (That is not the same as saying you must *never* launch a grid whose thread count is not divisible by 32.)

## 2. Memory hierarchy and latency

Throughput hides latency, but latency does not disappear. The GPU is built to switch among many threads; the physics of how far data sits from the SIMD unit still dominates when you stall.

The table below is order-of-magnitude for Ampere/Hopper-class GPUs. Actual cycle counts move with SKU, clock, cache hit/miss behavior, and access pattern—treat this as intuition, not a spec sheet to memorize.

| Storage | GPU cycles (typical ballpark) | Location | Notes |
| ---- | --- | ---- | ---- |
| Registers | ~0–1 | Inside SM | Fastest, but a finite pool; watch RAW hazards—often 20+ cycles between dependent FP ops without enough ILP to hide. |
| Shared memory / L1 | ~20–30 | Inside SM | Programmable fast memory; watch bank conflicts. |
| L2 | ~200 | Shared on die | Last large on-chip buffer before DRAM. |
| Global (DRAM) | ~400–800+ | HBM/GDDR | Often the first-order bottleneck. |

Other numbers worth anchoring:

- 32 bytes: on modern NVIDIA GPUs, L2 often tracks 32-byte sectors as a relatively fine-grained unit of fetch/tagging. Full cache line widths vary by level and architecture—do not assume “always 128-byte line = four sectors”—but scattered traffic still tends to pull data in sector-sized chunks, so coalescing, alignment, and locality stay critical.
- 128 bytes: a warp doing a clean, contiguous load of `float`s often looks like 32 × 4 B = 128 B at the programming model; underneath, that may be a small number of efficient 32 B sector transactions rather than one mythical monolithic 128 B “atomic” op.
- 100+ (warps in the neighborhood): to hide hundreds of cycles of global memory latency on a miss, you generally need a lot of concurrency that can actually make progress while others wait—how many warps “enough” depends on arithmetic intensity, ILP, occupancy limits, and architecture. Slogans like “hundreds of warps per SM” are less reliable than Nsight Compute telling you *why* you stalled.

## 3. Shared memory: 32 banks, 4-byte width (typical)

For common 32-bit accesses (e.g. `float`), think of shared memory as 32 banks with 4-byte strided mapping (the exact formula is in the *CUDA C Programming Guide* for your compute capability—wider modes and special cases change the conflict picture).

- Bank conflict: multiple lanes in the same warp hit the same bank but different addresses within it; the hardware serializes pieces of that access, which shows up as extra latency and lower effective bandwidth.
- Worst case: a 32-way conflict is the mental image of “one warp load chopped into ~32 rounds of service” (real cycles depend on the instruction mix and microarchitecture; the division model is still a useful qualitative anchor).

$$
\text{Effective Bandwidth} \approx \frac{\text{Peak Bandwidth}}{\text{Conflict Degree}}
$$

## 4. Data movement: PCIe vs. NVLink

Data motion is a performance killer. Any Host (CPU) ↔ Device (GPU) hop pays dearly compared to on-device bandwidth.

- PCIe Gen4 x16: one-way effective throughput is on the order of 31–32 GB/s (16 GT/s × 16 lanes, after 128b/130b encoding). Marketing often quotes ~64 GB/s bidirectional aggregate—do not confuse that with 64 GB/s each way.
  - Measured goodput often sits below the raw link math: protocol overhead, pinned vs. pageable host memory (`cudaMallocHost` / `cudaHostRegister`), async pipelining, copy granularity and TLP shape, and Root Complex / CPU topology all matter. ~25–30 GB/s one-way is common on many desktops and workstations; if you are lower, check whether the link degraded to ×8, whether traffic went through the PCH, and whether the copy path is sane.
- PCIe Gen5 x16: one-way effective throughput is on the order of 63–64 GB/s; bidirectional marketing sums to ~128 GB/s (same “do not mix up with one-way” caveat).
- NVLink (illustrative: H100-class systems): public materials often cite ~900 GB/s-class numbers for aggregate GPU–GPU interconnect in a given topology; for a single link or routable path in *your* box, use the whitepaper and system specs.
- Device memory bandwidth (H100 HBM3 class): peak marketing is commonly around 3.35 TB/s (3350 GB/s) for some SKUs—still distinct from what you sustain thermally and in real kernels.

The performance takeaway: dividing HBM peak by PCIe effective bandwidth gives you a small ratio—often single-digit percent. Treat that as a warning about round trips and host/device ping-pong: fuse when you can, keep state on device when you can, and overlap copies with compute where the programming model allows. Sometimes leaving work on the GPU—even if the GPU is not the theoretically “best” device for that step—still wins over PCIe churn (always weigh accuracy, libraries, and engineering constraints).

## 5. Kernel launch overhead: microseconds, not nanoseconds

Submitting work from the host is not free. Getting a kernel into the GPU’s queue is often on the order of a few microseconds (empirically ~3–10 µs is a common band, but cold start, power states, how you synchronize, and stream reuse all move the number; CUDA Graphs can amortize submission to something much smaller per replay).

- If the kernel body is only ~1 µs or less, launch and scheduling can dominate end-to-end time.
- Mitigations: CUDA Graphs, kernel fusion (fewer, bigger launches), avoid gratuitous synchronization, and batch tiny problems so each launch “looks like” one large piece of work.

## Summary: a quick checklist

Before you write the next line of CUDA, ask:

- Does my grid/block configuration expose enough warps / occupancy to hide memory and instruction latency? (Do not fetishize “32”—watch live warps and register/shared limits.)
- Are my global loads coalesced and vector-friendly? (Typical warp patterns tie to 32 × 4 B = 128 B and 32 B sectors; let Nsight Compute be the source of truth on memory metrics.)
- Am I incurring shared memory bank conflicts? (32 banks, 4-byte typical width.)
- Am I shuttling small payloads over PCIe? (Bandwidth and latency traps.)
- Is my kernel large enough relative to launch overhead?
