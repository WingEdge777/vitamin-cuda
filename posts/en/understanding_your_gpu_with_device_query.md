# A Deep Dive into DeviceQuery: Understanding Your GPU Hardware

## 0. Preface

Before writing a single line of high-performance CUDA code, you must know your silicon. deviceQuery is often the first command a developer runs, yet its output is usually ignored. This post translates those raw hardware limits into actionable programming mental models.
(Some familiarity with CUDA programming basics is assumed — concepts like compute capability, warps, blocks, and grids.)

The complete source code for all kernels and performance benchmarks discussed in this post is available in my open-source practice project: [WingEdge777/Vitamin-CUDA](https://github.com/WingEdge777/Vitamin-CUDA)

## 1. DeviceQuery Output

Using my **RTX 5060** Laptop GPU as an example.

```bash
cd samples/deviceQuery && bash run.sh
```

### 1.1 Output

```yaml
:: 1 available devices

  CUDA Driver Version / Runtime Version          13.1 / 12.9
  CUDA Capability Major/Minor version number:    12.0
  Total amount of global memory:                 8151 MBytes (8546484224 bytes)
  (026) Multiprocessors, (128) CUDA Cores/MP:    3328 CUDA Cores
  GPU Max Clock rate:                            1455 MHz (1.46 GHz)
  Memory Clock rate:                             12001 MHz
  Memory Bus Width:                              128-bit
  L2 Cache Size:                                 33554432 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        102400 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      No
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::CUDASetDevice() with device simultaneously) >
```

## 2. Detailed Breakdown

This is an RTX 5060. While the specs are all listed above, what matters to developers is understanding the **performance bottlenecks and optimization opportunities** behind these numbers.

### 2.1 Architecture Basics

- CUDA Driver / Runtime Version: 13.1 / 12.9
- Compute Capability: 12.0 (Blackwell architecture)
  - The architecture determines the supported instruction set. Higher compute capabilities generally mean better low-precision compute. Blackwell natively supports FP4, FP8, BF16, INT8, and other low-precision formats — ideal for quantized LLM inference.

### 2.2 Compute Power & Concurrency

- Streaming Multiprocessors (SMs): 26
  - CUDA Cores per SM: 128
    - This means 128 threads can truly execute in parallel per SM.
  - Total CUDA Cores: 3,328 (26 × 128)
    - At the hardware level, at most 3,328 threads are physically executing simultaneously.
  - **Programming takeaway:** Although there are only ~3K physical cores, the GPU's design philosophy — **latency hiding through massive parallelism** — means we must launch far more threads than that.
  - **Grid sizing:** The number of blocks should ideally be a multiple of the SM count (26), with several hundred or more blocks total, so the scheduler can rotate blocks effectively to hide latency.
- Max resident threads per SM: 1,536
- Max threads per block: 1,024
  - **Occupancy calculation:**
    - 1,024 threads/block: only 1 block fits per SM (1024 < 1536, but 2 blocks would exceed it). Utilization = 1024/1536 ≈ 66%.
    - 512 threads/block: 3 blocks fit per SM. Utilization = 1536/1536 = 100%.
  - **Sweet spot:** 128 or 256 threads/block is a solid general-purpose choice — easy to achieve high occupancy while balancing register and shared memory allocation. For peak optimization, run the numbers through the Occupancy Calculator.
  - Max active threads across the entire GPU: 1,536 × 26 = 39,936.

### 2.3 Clock Frequencies

- GPU Max Clock: 1,455 MHz (1.46 GHz)
  - ~0.5 ns to execute the simplest instruction (e.g., FADD).
  - **CPU comparison:** My Intel Ultra 9 285H boosts to 2.9 GHz+. Clock-for-clock, combined with branch prediction and out-of-order execution, GPUs are terrible at serial logic.
  - **Design philosophy:** The GPU embraces **strength in numbers**. It doesn't chase single-thread speed — instead, it pushes tens of thousands of threads (39,936) forward simultaneously.
  - **Programming takeaway:** Never write complex serial logic or deeply nested if/else in kernels.
- Memory Clock: 12,001 MHz
  - This is the effective frequency; the actual calculation involves Command Clock, Write Clock, DDR, etc. For development purposes, we only care about the resulting bandwidth throughput.
  - **Latency vs. Throughput:** A high clock doesn't mean low latency. A single DRAM access can take hundreds of GPU clock cycles. CUDA programs are almost always memory-bound (especially on this card). The core optimization goal is always **hiding that latency**.

### 2.4 Memory Subsystem (The Source of Performance Bottlenecks)

- Total VRAM: 8 GB
- Memory Bus Width: 128 bits
  - This is quite narrow. For reference, the A100 has 5,120 bits (128 × 40 HBM stacks). We can compute the bandwidth: 12001 × 2 × 128 / 8 = 384,000 MB/s ≈ **384 GB/s**. Our kernels will almost certainly be bandwidth-bound. If a kernel processing large data doesn't approach this throughput, there's room for improvement (and L2 cache can boost effective bandwidth further).
  - **Reality:** 128-bit bus width is very narrow. Kernels on this card are overwhelmingly bandwidth-limited.
  - **Coalesced access:** While physical DRAM has burst sizes, from the CUDA core's perspective, the minimum granularity for global memory access is a 32-byte **sector**.
    - Good: 32 threads read 128 contiguous bytes (e.g., one `float` each), coalesced into 4 sector transactions.
    - Bad: Reading 1 byte, or strided access — the bus still moves an entire 32-byte sector, wasting bandwidth.
  - **Programming takeaway:** Adjacent threads must access adjacent addresses (spatial locality). Use vectorized loads/stores (`float4`) whenever possible to reduce instruction count and improve ILP.
- L2 Cache: 32 MB
  - L2 also manages data in 32-byte sectors (cache lines are typically 128 bytes).
  - **Spatial locality:** L2 is shared across the entire GPU and much faster than VRAM. Exploit spatial locality — threads within a warp should access neighboring data to maximize L2 hit rate. For a narrow-bus card like the RTX 5060, the generous 32 MB L2 is a lifeline.
- Constant Memory: 65,536 bytes (64 KB)
  - Read-only, shared across all threads. Used for constants like weights. Faster than global memory thanks to a dedicated cache.

### 2.5 Resource Limits & Occupancy

- Max Shared Memory per SM: 100 KB
- Max Shared Memory per Block: 48 KB
  - **Gotcha:** If one block allocates 48 KB of SMEM, an SM can only run 2 blocks (48 × 2 < 100), potentially leaving the SM underutilized and lowering occupancy.
- Register File per Block: 65,536 registers
  - Beyond SMEM limits, per-thread register usage also caps the number of active threads/blocks, affecting occupancy.
  - **Register spill:** Registers are a scarce resource. If a single thread uses too many, the number of concurrent blocks per SM decreases.
  - **Worst case:** If registers are truly exhausted, the compiler spills variables to **local memory** — which physically resides in VRAM. This is extremely slow and will devastate your pipeline.
- **Bank Conflicts:** Shared memory is divided into 32 banks. Within a warp, if multiple threads simultaneously access different addresses in the same bank, the accesses are serialized. Banks are assigned in 4-byte (32-bit) round-robin: addresses 0–31 map to banks 0–31, then 32–63 map to banks 0–31 again, and so on.
  - **Guideline:** The simplest approach is to have adjacent threads access adjacent addresses (`tid` → `data[tid]`), which is naturally conflict-free. For more complex index transformations (e.g., swizzled access patterns for tiled algorithms), conflicts must be carefully avoided.

### 2.6 Scheduling & Physical Limits

- Warp size: 32
  - The warp is the GPU's minimum scheduling unit. Each warp contains 32 threads executing the same instruction in lockstep — the foundation of GPU parallelism.
    - A block of 256 threads is dispatched to an SM as 8 warps, scheduled in groups of 32.
  - **Warp divergence:** Under the SIMT model, if threads hit an if/else branch, both paths execute serially — threads outside the active branch are masked.
- Max block dimensions (x, y, z): (1024, 1024, 64)
  - Block dimensions are capped (the total still cannot exceed 1,024 threads).
  - Multi-dimensional blocks are purely a convenience for indexing data in multiple dimensions.
- Max grid dimensions (x, y, z): (2,147,483,647, 65,535, 65,535)
  - In most cases, a 1D grid is sufficient (just don't overflow `int`). Use multi-dimensional grids only when the data naturally maps to multiple dimensions.
- Max memory pitch: 2,147,483,647 bytes
  - When allocating 2D pitched memory, the per-row stride cannot exceed `INT_MAX`.
- Texture alignment: 512 bytes
  - Texture memory addresses must be aligned to 512 bytes for efficient access.
- Async Copy:
  - Supports concurrent Copy Engine and Compute Engine operation (dual pipeline) — the hardware foundation for compute-transfer overlap.
- The remaining fields are self-explanatory.

## 3. Key Takeaways

- **Block size:** Use 128 or 256 as a default. Grid size should be at least several times the SM count (26) — ideally tens of times.
- **Memory access rules of thumb:**
  - **Coalesced:** Adjacent threads must access contiguous addresses.
  - **Vectorized:** Use `float4` over `float` whenever possible.
- **Latency hiding:** Don't be afraid of massive thread counts. Leverage thread-level parallelism to mask memory latency through warp scheduling.
- **Resource management:** Keep a close eye on SMEM and register usage. Prevent occupancy collapse or register spills to VRAM.

These are my personal insights on GPU architecture and CUDA programming. Feel free to star [Vitamin-CUDA](https://github.com/WingEdge777/Vitamin-CUDA) and join the discussion.
