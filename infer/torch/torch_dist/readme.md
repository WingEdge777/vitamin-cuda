# 使用 PyTorch 进行分布式应用开发 - 分布式推理入门实战

如今模型越来越大。当模型参数量达到数百亿级别，即使通过 INT4 量化压到单卡能装下权重，推理时的 KV Cache 和激活值也会随 batch size 和序列长度线性增长，单卡显存很快捉襟见肘——多卡分布式推理几乎是必经之路。

但想直接阅读 Megatron-LM、DeepSpeed、vllm、sglang 源码学习时，错综复杂的工程封装往往让人望而生畏。本文抛开所有工程包袱，仅用不到 200 行核心 PyTorch 代码，带你从零手搓大模型分布式的三大核心流派的实现。

本文使用一个简单的两层 Linear 模型，两个 GPU，手把手带你理解并实现三种主流并行推理模式：数据并行 (DP)、张量并行 (TP)、流水线并行 (PP)。同时了解基本的 `torch.distributed` 库的通信原语使用。

**环境要求**: PyTorch >= 2.0，2× NVIDIA GPU，NCCL backend。推荐使用 NVIDA 的 NGC PyTorch 镜像

## 目录

1. [环境准备与基础概念](#1-环境准备与基础概念)
2. [模型定义](#2-模型定义)
3. [数据并行 (DP)](#3-数据并行-dp)
4. [张量并行 (TP)](#4-张量并行-tp)
5. [流水线并行 (PP)](#5-流水线并行-pp)
6. [运行结果](#6-运行结果)
7. [三种并行对比](#7-三种并行对比)

---

## 1. 环境准备与基础概念

### 1.1 分布式初始化

```python
import os
import torch
import torch.distributed as dist

def setup_dist(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_dist():
    dist.destroy_process_group()
```

关键概念：

- **rank**: 进程编号，0 或 1
- **world_size**: 总进程数，本文固定为 2
- **NCCL**: NVIDIA 集合通信库，GPU 间通信的高效实现

### 1.2 进程启动

```python
import torch.multiprocessing as mp

# 主进程初始化模型和数据（保持在 CPU 上，避免 CUDA 跨进程问题）
torch.manual_seed(42)
model = TwoLayerModel(hidden_size)
x = torch.randn(batch_size, hidden_size)

# 计算参考输出
with torch.inference_mode():
    ref_output = model(x)

# 启动多进程，rank 自动作为第一个参数传入
mp.spawn(run_demo, args=(world_size, model, x, ref_output), nprocs=world_size, join=True)
```

> **注意**: 模型和数据必须保留在 CPU 上创建，由各子进程自行 `.cuda()` 到对应设备。`mp.spawn` 创建的子进程无法直接共享主进程的 CUDA 上下文。

### 1.3 通信原语

本文用到的通信原语：

| 原语 | 类型 | 说明 |
|------|------|------|
| all_gather | 集合通信 | 每个 rank 收集所有 rank 的数据 |
| all_reduce | 集合通信 | 所有 rank 的数据做规约操作（如求和） |
| isend/irecv | 点对点通信 | 异步发送/接收数据 |
| batch_isend_irecv | 点对点通信 | 批量提交多个异步 send/recv 操作 |

```text
all_gather:
Rank 0: [a] ──┐
              ├──> [[a], [b]] （所有 rank 都有）
Rank 1: [b] ──┘

all_reduce(SUM):
Rank 0: 10 ──┐
             ├──> 30 （所有 rank 都有）
Rank 1: 20 ──┘

isend/irecv:
Rank 0               Rank 1
  │                    │
  ├─── isend ─────────>│
  │<──────── irecv ────┤
  │                    │
```

---

## 2. 模型定义

```python
class TwoLayerModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.Linear(hidden_size, hidden_size, bias=False),
        ])

    def forward(self, x):
        x = torch.relu(self.layers[0](x))
        x = self.layers[1](x)
        return x
```

使用 `ModuleList` 便于后续切分不同层。同时设置 `bias=False` 是为了在演示张量并行（TP）时，无需额外处理偏置项的切分逻辑，从而降低入门的理解成本。

---

## 3. 数据并行 (DP)

### 3.1 原理

```text
       输入 [B, H]
            │
      ┌─────┴─────┐
      ▼           ▼
┌──────────┐ ┌──────────┐
│  Rank 0  │ │  Rank 1  │
│ 完整模型 │ │ 完整模型 │
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
      [B, H] 输出
```

每个 GPU 持有完整模型副本，数据按 batch 切分，通过 `all_gather` 收集结果。

### 3.2 代码实现

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

要点：
- 每个 rank 只取自己负责的 batch 切片进行计算
- `gathered` 缓冲区在首次 `forward` 时按实际输出形状动态分配
- `all_gather` 后拼接得到完整输出

> **训练 vs 推理**: 本文的 DP 聚焦推理场景，核心是 `all_gather` 收集各 rank 的输出。训练场景下的 DP（如 `DistributedDataParallel`）核心则是 `all_reduce` 同步梯度。

### 3.3 all_gather

```text
Rank 0: [a, b] ──┐
                 ├──> all_gather ──> [[a,b], [c,d]] （两个 rank 都有）
Rank 1: [c, d] ──┘
```

---

## 4. 张量并行 (TP)

### 4.1 原理

Column Parallel: 权重按列切分，输出在 hidden 维度分片
Row Parallel: 权重按行切分，输入 hidden 维度分片，all_reduce 聚合

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
X [B,H] ─切分─> X_0 [B,H/2]    X_1 [B,H/2]
                    │              │
                    ▼              ▼
               X_0 @ W_0^T    X_1 @ W_1^T
                    │              │
                    └──────┬───────┘
                           ▼
                    all_reduce(SUM)
                           │
                           ▼
                      [B, H] 输出
```

Column + Row 组合是 TP 的精髓所在：Column Parallel 的输出 `[B, H/2]` 恰好是 Row Parallel 需要的分片输入，两者天然衔接，中间无需任何通信，只在最后 `all_reduce` 一次。

在真实的 LLM 中，Column Parallel 和 Row Parallel 通常被应用于 Attention 模块的 QKV linear 投影 和 Out linear 投影。在做切分时，通常是按注意力头的数量进行切分，保证每个 GPU 独立计算一部分头，最后再 all_reduce 汇总。

### 4.2 代码实现

```python
class ColumnParallelLinear(nn.Module):
    def __init__(self, full_linear):
        super().__init__()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        out_per_rank = full_linear.out_features // world_size
        self.weight = nn.Parameter(full_linear.weight.detach()[rank*out_per_rank : (rank+1)*out_per_rank].clone())

    def forward(self, x):
        return x @ self.weight.t()

class RowParallelLinear(nn.Module):
    def __init__(self, full_linear):
        super().__init__()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        in_per_rank = full_linear.in_features // world_size
        self.weight = nn.Parameter(full_linear.weight.detach()[:, rank*in_per_rank : (rank+1)*in_per_rank].clone())

    def forward(self, x):
        out = x @ self.weight.t()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

class TPModel(nn.Module):
    def __init__(self, full_model):
        super().__init__()
        self.fc1 = ColumnParallelLinear(full_model.layers[0])
        self.fc2 = RowParallelLinear(full_model.layers[1])

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
```

### 4.3 all_reduce

```text
Rank 0: 10 ──┐
             ├──> all_reduce(SUM) ──> 30 （两个 rank 都有）
Rank 1: 20 ──┘
```

---

## 5. 流水线并行 (PP)

### 5.1 原理

```text
时间 T0:
┌──────────┐    ┌──────────┐
│  Rank 0  │    │  Rank 1  │
│ Layer 0  │    │  (等待)  │
│  计算中  │    │          │
└────┬─────┘    └──────────┘
     │ isend
     ▼
   激活值

时间 T1:
┌──────────┐    ┌──────────┐
│  Rank 0  │    │  Rank 1  │
│  (等待)  │    │ Layer 1  │
│          │    │  计算中  │
└──────────┘    └────┬─────┘
     ▲               │ isend
     │               ▼
     └──irecv     最终输出
```

Layer 切分到不同 GPU，数据依次流过各阶段，使用点对点通信传输激活值。

如图所示，当 Rank 0 在计算时，Rank 1 处于等待状态，这种算力闲置的真空期被称为流水线气泡 (Pipeline Bubble)。两阶段 PP 的理论气泡率 ≈ 50%（一半时间有 GPU 空闲），这也解释了为什么运行结果中 PP 的延迟远高于 TP。工业界通常引入微批次 (Micro-batch) 配合 1F1B 调度来压缩气泡。例如，不需要等一整批数据（Batch）全走完 Layer 0 才传给 Layer 1。而是把 Batch 拆成更小的 Micro-batch，Layer 0 处理完第一个小块就赶紧发过去，让下游 GPU 先转起来。

### 5.2 代码实现

```python
class PPModel(nn.Module):
    def __init__(self, full_model):
        super().__init__()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        self.rank = rank
        self.world_size = world_size
        self.is_first = (rank == 0)
        self.is_last = (rank == world_size - 1)
        self.total_layers = len(full_model.layers)

        layers_per_rank = len(full_model.layers) // world_size
        self.start_idx = rank * layers_per_rank
        self.layers = nn.ModuleList([
            full_model.layers[i] for i in range(self.start_idx, (rank+1) * layers_per_rank)
        ])

        self.recv_buffer = None

    def _send(self, tensor, dst):
        reqs = dist.batch_isend_irecv([dist.P2POp(dist.isend, tensor.contiguous(), dst)])
        for req in reqs:
            req.wait()

    def _recv(self, src):
        reqs = dist.batch_isend_irecv([dist.P2POp(dist.irecv, self.recv_buffer, src)])
        for req in reqs:
            req.wait()
        return self.recv_buffer

    def forward(self, x):
        if self.recv_buffer is None:
            self.recv_buffer = torch.empty_like(x)

        # 1. 接收阶段
        # 首 rank 使用传入的 x，其余 rank 忽略 x，从前一个 rank 接收激活值
        if self.is_first:
            out = x
        else:
            out = self._recv(self.rank - 1)

        # 2. 计算阶段
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if self.start_idx + i < self.total_layers - 1:
                out = torch.relu(out)

        # 3. 发送阶段
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

### 5.3 为什么用 batch_isend_irecv

你可能会问：为什么不直接用 `dist.send()` / `dist.recv()`？

1. **NCCL 不支持同步 P2P**：`dist.send()` / `dist.recv()` 是同步阻塞 API，仅 Gloo 后端可用。NCCL 后端只支持异步的 `isend` / `irecv`。
2. **官方推荐的 P2P 方式**：`batch_isend_irecv` 是 PyTorch 官方推荐的 NCCL 点对点通信 API。虽然本文每次只提交一个 op，但在更复杂的流水线场景中（如多微批次 1F1B 调度），一个 rank 可能需要同时发送和接收不同微批次的数据，此时将多个 P2P 操作打包提交可以避免因调用顺序不一致导致的死锁。

```python
# _send / _recv 封装了 batch_isend_irecv 的样板代码
def _send(self, tensor, dst):
    reqs = dist.batch_isend_irecv([dist.P2POp(dist.isend, tensor.contiguous(), dst)])
    for req in reqs:
        req.wait()

def _recv(self, src):
    reqs = dist.batch_isend_irecv([dist.P2POp(dist.irecv, self.recv_buffer, src)])
    for req in reqs:
        req.wait()
    return self.recv_buffer
```

---

## 6. 运行结果

```bash
python test.py
```

```text
==================================================
Demo 0: Single GPU
==================================================
[Single GPU] Output match reference: True, Avg time: 0.094 ms

==================================================
Demo 1: DP
==================================================
[DP] Output match reference: True, Avg time: 0.101 ms

==================================================
Demo 2: TP
==================================================
[TP] Output match reference: True, Avg time: 0.067 ms

==================================================
Demo 3: PP
==================================================
[PP] Output match reference: True, Avg time: 0.251 ms
```

---

## 7. 三种并行对比

| 维度 | DP | TP | PP |
|------|-----|-----|-----|
| 切分对象 | 数据 | 模型参数 | 模型层 |
| 模型副本 | 完整 | 部分参数 | 部分层 |
| 内存占用 | 高 | 低 | 低 |
| 通信算子 | all_gather | all_reduce | isend/irecv |
| 通信频率 | 每次推理一次 | 每组 Column+Row 一次 | 每阶段一次 |
| 适用场景 | 小模型大 batch | 大 hidden | 超深模型 |
| 延迟特点 | 低延迟 | 低延迟 | 存在 bubble |

选择建议：

- **DP**: 模型能单卡放下，需要高吞吐
- **TP**: 单层参数太大，追求低延迟
- **PP**: 层数太多，可接受一定延迟

**💡 LLM 推理的阶段特性：**
LLM 推理特有的两个阶段。Prefill（预填充） 阶段是计算密集型的，大 Batch 进来时，TP 能有效降低首字延迟（TTFT）；而 Decode（解码） 阶段是访存密集型的，此时按 TP 切分不仅切分了计算，也相当于把庞大的 KV Cache 分摊到了多张卡的显存中。

实际大模型推理通常组合使用：TP + PP 处理超大模型，DP 提升吞吐。理解了这三种基础范式，再去阅读 Megatron-LM/DeepSpeed 或 vllm/sglang 源码，就不再是看天书了。

# 8. 结束

本文为了聚焦核心逻辑，所有代码均在单机双卡环境下运行。但在真实的工业界场景中，通常通过环境变量配置结合 `torchrun` 即可利用 NCCL 完成多机多卡推理。需要注意的是，多机环境往往还需要 RDMA 网络（如 InfiniBand 或 RoCE）的硬件支持来实现低时延的数据交换，不过这部分超出了本入门教程的范围，所以没有加以讨论。

掌握了分布式推理的宏观通信原语，仅仅是第一步。当张量被切分到单张 GPU 上后，如何把单卡硬件的性能压榨到极致？如何写出比 PyTorch 原生算子更快、显存占用更低的手写 CUDA Kernel？

实际上，将高性能的自研 Kernel 包装成 PyTorch Ops，正是当代顶级推理框架（如 vLLM, SGLang）的标准做法。如果你对底层的极致性能优化感兴趣，欢迎浏览本人的硬核算子库项目 **[`vitamin-cuda`](https://github.com/你的链接)**，里面包含了上百个手写 CUDA 算子。欢迎共同学习、交流与进步！

完整代码见 [test.py](./test.py)

以上
