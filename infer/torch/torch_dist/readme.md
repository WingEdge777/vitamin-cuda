# Distributed Inference with PyTorch from First Principles: DP, TP, and PP in Less Than 200 Lines
>
> Note: Text translated by AI. Code crafted by human.
>

## 0. Preface

Models keep getting bigger. Even if INT4 quantization squeezes the weights onto a single GPU, inference still has to pay for KV cache and activations, both of which scale with batch size and sequence length. In practice, single-GPU inference quickly hits the VRAM wall. Multi-GPU distributed inference is no longer optional.

The problem is that if you jump straight into Megatron-LM, DeepSpeed, vLLM, or SGLang source code, the engineering layers can bury the core ideas. So this note does the opposite: strip everything down to the bare minimum and rebuild the three canonical inference parallelism strategies directly with PyTorch distributed primitives.

The demo uses:

- a tiny two-layer `Linear` model
- 2 GPUs
- less than 200 lines of core code

From that, we build and benchmark:

- Data Parallelism (DP)
- Tensor Parallelism (TP)
- Pipeline Parallelism (PP)

Full code: [vitamin-cuda torch_dist]((https://github.com/WingEdge777/vitamin-cuda/tree/main/infer/torch/torch_dist))

Environment:

- PyTorch >= 2.0
- 2 NVIDIA GPUs
- NCCL backend

I recommend using NVIDIA's NGC PyTorch container if you want the least friction.

## 1. Distributed Setup and Communication Basics

### 1.1 Process Group Initialization

```python
import os
import torch
import torch.distributed as dist

def setup_dist(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=rank)
    torch.cuda.set_device(rank)

def cleanup_dist():
    dist.destroy_process_group()
```

Three terms matter here:

- `rank`: process id, either `0` or `1` in this demo
- `world_size`: total number of processes, fixed to `2`
- `NCCL`: NVIDIA's high-performance communication backend for GPU collectives and P2P

### 1.2 Launching Multiple Processes

```python
import torch.multiprocessing as mp

torch.manual_seed(42)
model = TwoLayerModel(hidden_size)
x = torch.randn(batch_size, hidden_size)

with torch.inference_mode():
    ref = model(x)

mp.spawn(
    run_demo,
    args=(cls, label, world_size, model, x, ref),
    nprocs=world_size,
    join=True,
)
```

One important detail: create the model and input on CPU in the parent process, then move them to GPU inside each child process. Sharing CUDA state across `mp.spawn` workers is the wrong mental model and usually ends badly.

### 1.3 Communication Primitives Used in This Demo

| Primitive | Type | Purpose |
| --- | --- | --- |
| `all_gather` | collective | gather shards from all ranks |
| `all_reduce` | collective | reduce values across all ranks, then broadcast result |
| `isend` / `irecv` | point-to-point | asynchronous send and receive |
| `batch_isend_irecv` | point-to-point | submit multiple P2P ops in one call |

```text
all_gather:
Rank 0: [a] ──┐
              ├──> [[a], [b]] on every rank
Rank 1: [b] ──┘

all_reduce(SUM):
Rank 0: 10 ──┐
             ├──> 30 on every rank
Rank 1: 20 ──┘

isend / irecv:
Rank 0               Rank 1
  │                    │
  ├─── isend ─────────>│
  │<──────── irecv ────┤
  │                    │
```

## 2. The Toy Model

```python
class TwoLayerModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(hidden_size, hidden_size, bias=False),
                nn.Linear(hidden_size, hidden_size, bias=False),
            ]
        )

    def forward(self, x):
        x = torch.relu(self.layers[0](x))
        return self.layers[1](x)
```

I use `ModuleList` so later we can slice layers cleanly for pipeline parallelism. I also set `bias=False` so tensor parallelism stays focused on the main weight sharding logic instead of bias bookkeeping.

## 3. Data Parallelism (DP)

### 3.1 Core Idea

Each GPU holds a full copy of the model. The batch is split across ranks. Every rank computes on its own shard, then all partial outputs are gathered back together.

```text
        Input [B, H]
             │
       ┌─────┴─────┐
       ▼           ▼
┌──────────┐ ┌──────────┐
│  Rank 0  │ │  Rank 1  │
│ full net │ │ full net │
│ Batch 0  │ │ Batch 1  │
└────┬─────┘ └────┬─────┘
     │            │
     ▼            ▼
 [B/2, H]     [B/2, H]
     │            │
     └─────┬──────┘
           ▼
      all_gather
           │
           ▼
       Output [B, H]
```

### 3.2 Implementation

```python
class DPModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.gathered = None

    def forward(self, x):
        local_batch = x.shape[0] // self.world_size
        start_idx = self.rank * local_batch
        end_idx = (self.rank + 1) * local_batch
        local_x = x[start_idx:end_idx]
        out = self.model(local_x)
        if self.gathered is None:
            self.gathered = [torch.empty_like(out) for _ in range(self.world_size)]
        dist.all_gather(self.gathered, out)
        return torch.cat(self.gathered, dim=0)
```

What matters:

- each rank only computes its own batch slice
- output buffers are allocated lazily from the real output shape
- `all_gather` reconstructs the full batch output

For inference, this is the right mental model for DP.
For training, classic `DistributedDataParallel` is a different story: the key communication becomes gradient `all_reduce`, not output `all_gather`.

## 4. Tensor Parallelism (TP)

### 4.1 Core Idea

Tensor parallelism splits the weights themselves.

- Column Parallel Linear: shard output features
- Row Parallel Linear: shard input features, then `all_reduce` the partial sums

```text
Column Parallel Linear:
X [B,H]      W [H,H]
                │
        ┌───────┴───────┐
        ▼               ▼
   W_0 [H/2,H]     W_1 [H/2,H]
        │               │
        ▼               ▼
    X @ W_0^T       X @ W_1^T
     [B,H/2]         [B,H/2]

Row Parallel Linear:
X [B,H] ─split─> X_0 [B,H/2]    X_1 [B,H/2]
                    │              │
                    ▼              ▼
               X_0 @ W_0^T    X_1 @ W_1^T
                    │              │
                    └──────┬───────┘
                           ▼
                    all_reduce(SUM)
                           │
                           ▼
                      Output [B, H]
```

The elegant part is the composition:

- column-parallel layer produces `[B, H/2]` on each rank
- row-parallel layer naturally consumes exactly that shard
- no communication is needed between the two layers
- only the final `all_reduce` is required

This is the same structural pattern used in real LLM blocks, especially around QKV projections and output projections.

### 4.2 Implementation

```python
class ColumnParallelLinear(nn.Module):
    def __init__(self, linear):
        super().__init__()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        out_per_rank = linear.out_features // world_size
        self.weight = nn.Parameter(
            linear.weight.detach()[
                rank * out_per_rank : (rank + 1) * out_per_rank
            ].clone()
        )

    def forward(self, x):
        return x @ self.weight.t()

class RowParallelLinear(nn.Module):
    def __init__(self, linear):
        super().__init__()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        in_per_rank = linear.in_features // world_size
        self.weight = nn.Parameter(
            linear.weight.detach()[
                :, rank * in_per_rank : (rank + 1) * in_per_rank
            ].clone()
        )

    def forward(self, x):
        out = x @ self.weight.t()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

class TPModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fc1 = ColumnParallelLinear(model.layers[0])
        self.fc2 = RowParallelLinear(model.layers[1])

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))
```

If you only remember one thing from this section, remember this:
**column-parallel + row-parallel is the canonical TP pairing.**

## 5. Pipeline Parallelism (PP)

### 5.1 Core Idea

Pipeline parallelism splits the model by layers. Activations flow from one rank to the next like a production line.

```text
Time T0:
┌──────────┐    ┌──────────┐
│  Rank 0  │    │  Rank 1  │
│ Layer 0  │    │  idle    │
│ compute  │    │          │
└────┬─────┘    └──────────┘
     │ isend
     ▼
 activation

Time T1:
┌──────────┐    ┌──────────┐
│  Rank 0  │    │  Rank 1  │
│  idle    │    │ Layer 1  │
│          │    │ compute  │
└──────────┘    └────┬─────┘
     ▲               │ isend
     │               ▼
     └── irecv   final output
```

This is simple to understand, but it comes with a cost: pipeline bubbles.

While rank 0 is busy, rank 1 waits.
Then rank 1 is busy, while rank 0 waits.
With only two stages, the idle fraction can be painfully obvious, which is exactly why PP often shows worse latency than TP in small demos like this one.

In real systems, people attack the bubble with micro-batching:

- training side: schedules like `1F1B`
- inference side: chunked prefill or continuous forward micro-batches

The principle is the same: keep every stage fed.

### 5.2 Implementation

```python
class PPModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        self.rank = rank
        self.world_size = world_size
        self.is_first = rank == 0
        self.is_last = rank == world_size - 1

        layers_per_rank = len(model.layers) // world_size
        start_layer = rank * layers_per_rank
        end_layer = (rank + 1) * layers_per_rank
        self.layers = nn.ModuleList(model.layers[start_layer:end_layer])

        self.start_idx = start_layer
        self.total_layers = len(model.layers)
        self.buf = None

    def _send(self, tensor, dst):
        reqs = dist.batch_isend_irecv(
            [dist.P2POp(dist.isend, tensor.contiguous(), dst)]
        )
        for req in reqs:
            req.wait()

    def _recv(self, src):
        reqs = dist.batch_isend_irecv([dist.P2POp(dist.irecv, self.buf, src)])
        for req in reqs:
            req.wait()
        return self.buf

    def forward(self, x):
        if self.buf is None:
            self.buf = torch.empty_like(x)

        if self.is_first:
            out = x
        else:
            out = self._recv(self.rank - 1)

        for i, layer in enumerate(self.layers):
            out = layer(out)
            if self.start_idx + i < self.total_layers - 1:
                out = torch.relu(out)

        if self.is_first:
            self._send(out, self.rank + 1)
            return self._recv(self.world_size - 1)
        elif self.is_last:
            self._send(out, 0)
            return None
        else:
            self._send(out, self.rank + 1)
            return None
```

### 5.3 Why `batch_isend_irecv` Instead of `dist.send()` / `dist.recv()`?

Two reasons:

1. NCCL does not support the blocking `send` / `recv` API path the way people often expect.
2. `batch_isend_irecv` is the PyTorch-recommended NCCL P2P interface, and it scales better to real pipeline schedules where one rank may need to post multiple sends and receives without deadlocking.

Even though this demo submits only one op at a time, the wrapper keeps the code aligned with real-world practice.

## 6. Benchmark

Run:

```bash
python test.py
```

Sample output:

```text
==================================================
Demo 0: Single GPU
==================================================
[Single GPU] Output match reference: True, Avg time: 10.009 ms

==================================================
Demo 1: DP
==================================================
[DP] Output match reference: True, Avg time: 5.123 ms

==================================================
Demo 2: TP
==================================================
[TP] Output match reference: True, Avg time: 5.748 ms

==================================================
Demo 3: PP
==================================================
[PP] Output match reference: True, Avg time: 11.032 ms
```

The exact numbers will vary by GPU and software stack, but the trend is the interesting part:

- DP helps because the batch is split cleanly
- TP helps because compute and parameter storage are partitioned
- PP suffers because the toy example has obvious bubbles and no micro-batch overlap

## 7. DP vs. TP vs. PP

| Dimension | DP | TP | PP |
| --- | --- | --- | --- |
| What is sharded | data | parameters | layers |
| Model replica | full | partial | partial |
| Memory footprint | high | low | low |
| Main communication | `all_gather` | `all_reduce` | `isend` / `irecv` |
| Communication cadence | once per inference | once per TP reduction point | once per stage handoff |
| Best fit | smaller model, larger batch | wide hidden layers, tight latency | very deep models |
| Latency behavior | usually good | usually good | bubble-prone |

Practical rule of thumb:

- use DP when the model already fits and you want throughput
- use TP when one layer is too wide for one GPU
- use PP when the model depth forces a layer-wise split

For real LLM serving, these strategies are usually combined rather than used alone:

- `TP + PP` for giant models
- `DP` for higher request throughput

During prefill, TP is especially useful for lowering time-to-first-token.
During decode, TP also helps distribute the KV cache footprint across multiple GPUs, which is often just as important as splitting the math.

## 8. Final Notes

This demo is intentionally limited to a single node with 2 GPUs, because the goal is conceptual clarity, not cluster orchestration. In production, the same ideas extend to multi-node setups through `torchrun`, NCCL environment variables, and usually RDMA-capable networking such as InfiniBand or RoCE.

Understanding the communication pattern is only step one. Once tensors land on a single GPU, the next question is harder and more interesting: how do you push the hardware to its limit with custom kernels?

That is exactly what modern inference systems do. Frameworks like vLLM and SGLang do not stop at distributed scheduling. They also wrap highly optimized custom kernels as PyTorch ops.

If that direction interests you, browse Github [vitamin-cuda](https://github.com/WingEdge777/vitamin-cuda). This project is full of hand-written CUDA kernels and low-level experiments built for exactly that purpose.
