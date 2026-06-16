# 一文入门推理系统性能优化：流水线、计算通信 Overlap 与 Offload/Onload 实践

## 0. 序

相信大家都听到过训练端 pytorch FSDP，ZeRO 1/2/3, Deepspeed/Megatron 的 pipeline 优化、offloading 之类的， 推理端 vLLM、SGLang 等框架会提到 计算和通信重叠，零 CPU 开销调度（zero-overhead schedule, 当然这个有 CUDA graph 的巨大功劳），offloading 计算 等等。

听起来有那么一点高大上，但究其实质，就是流水线优化（streaming/pipelining），而且流水线优化的核心思路十分简单。本文将用不到 100 行核心 pytorch 代码，向大家阐述流水线优化的核心思想，展示计算和通信如何 overlap、streaming onload 的实操是什么样的。

## 1. 流水线优化 - Streaming/pipelining

为什么我有 流水线优化 作为入门系统性能优化这一说？因为这东西太常见了。大到上层业务开发的模块与微服务拆分，中到网络 IO 与用户程序优化，小到硬件底层的指令调度，几乎都能用得上。但奇怪的是，很少有书本或教程能把这些跨层级的概念打通并直白地阐述出来，当然，这可能和我读书少有关系。而我个人风格也是上手就干，哪里有问题就解决哪里。所以刚毕业工作时，我其实没有系统性优化的经验，最开始只知道打时间日志，后来会用 perf/profiler 打个 timeline trace/火焰图，找出最长的区间，然后优化它。

工作多年后，我觉得有必要给新人补上‘流水线优化’这一课。这东西太基础也太实用了，但在学校的理论体系和实际的工程应用之间，往往存在着巨大的断层。这也导致很多人在做系统分析时缺乏这种直觉，或者自己已经用上了却依然不明就里。

不管那些训推框架吹得多么天花乱坠，streaming 就是很简单的东西，它的核心思想是什么？很少有资料用大白话直接点明，我就大言不惭一下，streaming 的核心思想是利用异步/多线程技术，让理论上的性能瓶颈成为现实系统中的真实瓶颈，从而逼出系统整体性能的理论上限。

首先，肯定有人会说我这里把异步和多线程混为一谈了。确实，在应用代码的语境下，基于事件循环的异步和开几个物理线程是两码事。但如果我们把视角穿透应用层，看向系统底层和物理硬件呢？当你用‘异步’把工作交接出去后，是谁在替你负重前行？是网卡在独立收发，是 DMA 控制器在帮你搬运内存，是 GPU 的 Command Processor 在吞吐指令。这些独立的硬件执行单元，本质上就是在和你的 CPU 主线程并发工作。在真实的复杂系统里，没有什么绝对孤立的‘单线程’（当然，请写嵌入式 bootloader 和 BIOS 的大佬略过）用户程序永远是在和庞大的底层并发系统/硬件协同运行的。

其次，我为什么说 streaming 很简单？因为就像你知道做菜之前先按电饭煲烧饭（你开了一个流去烧饭，然后自己同时做菜），先按下洗衣机再去拖地（你开了一个流去洗衣服，然后自己同时拖地）一样。只要你会这些，恭喜你，你已经掌握 streaming 的用法了。

好，有点扯远了，说回后半句，什么叫 让理论上的性能瓶颈成为现实系统中的真实瓶颈？

学过数据结构、离散数学的肯定知道，在一组系统作业中（一组系统作业一般可以表现为有向无环图的形式），那整个系统的性能是由其中的关键路径决定的。

如下图，每个节点有名称和任务耗时：

![DAG](https://cdn.jsdelivr.net/gh/WingEdge777/CDN@main/images/vitamin_cuda/cpm.png)

什么是关键路径？就是从起点到终点累计耗时最长的那条路径，即图中红色节点构成的 S-->B-->C-->D-->F-->T。

我们知道，在串行系统中，有依赖的任务集合，只要按拓扑顺序执行一遍即可完成所有作业。但想把系统性能优化到理论上限，你得保证关键路径上的作业前后背靠背的紧贴着执行，其他作业用单独的线程/cpu 核/硬件去执行。这样，才能让理论上的性能瓶颈成为真实的系统瓶颈。至于如何对关键路径本身进行算法级的极限优化，则属于另一个维度的考量，超出了本文的讨论范畴，暂且按下不表。。

最后，流水线化具体怎么实现呢？这个问题其实没有标准答案，得具体 case 具体分析。没有人可以几句话说明如何流水线化。不过，基于我个人的工程实战直觉，大概可以归纳为核心的两点：

- 数据强依赖的任务放在一条流水线（同一个 Stream）：上一个任务的结果是下一个任务的输入，把它们放在同一个队列里，利用底层本身的 FIFO 特性来保证绝对的执行顺序，免去手动同步的开销。
- 无强数据依赖，且占用正交物理资源的任务，放到不同的流水线：比如 CPU 预处理与 GPU 计算、网络 IO 与磁盘读写、PCIe 权重 Copy 与 GPU 矩阵乘法。它们互不相干且消耗不同硬件模块的资源，把它们拆到不同流水线上，就能实现完美的 Overlap。

真实业务系统中 DAG 状态可能会非常复杂，什么时间点发起哪一个任务的执行，如何控制数据的流向，同步点的设置等等，都要仔细分析并实现。所以前文我说那些框架吹得天花乱坠，但其实在复杂的业务系统实现流水线优化并隐藏部分开销，还是很🐂🍺的。

而复杂系统流水线化后的程序代码通常都充斥着异步回调（嵌套不知道几层，又叫回调地狱）、多线程、N 个队列传输数据，所以读起来可能十分晦涩。

因此本文将用一个十分具体且简单的场景，24 层的 MLP 模型前向推理，在几乎不影响推理速度的前提下，仅使用两层权重 buffer + 原生 pytorch 多 stream 前向推理展示，节省 75%显存占用。这个 case 涉及了双流水线调度、copy 与 GPU 计算重叠，以及 Streaming Onload（权重按需加载至显存），完美点题。其实我们这个场景也展示了超大模型在显存受限推理场景的一种解决方案，就是 double buffer + stream loading + 逐层计算。

## 2. 串行 MLP 模型

本文用一个 24 层的 MLP 模型作为 baseline，线性层输入输出大小都为 8192（为了制造较大的计算负载以隐藏 copy 时延，教学演示用途）。baseline 很简单，就是初始化 model，to gpu，forward 即可（同理输入 batchsize 也用 8192）。
代码如下：

```python
import torch
import torch.nn as nn
import math

MAX_BATCH_SIZE = 8192
FEAT_DIM = 8192
NUM_LAYERS = 24
DTYPE = torch.bfloat16  # 采用 BF16

class BaseMLPModel(nn.Module):
    """base model"""

    def __init__(self, num_layers, cpu_weights, device):
        super().__init__()
        self.device = device
        self.num_layers = num_layers
        self.layers = nn.ParameterList(
            [nn.Parameter(w.clone().detach().to(self.device)) for w in cpu_weights]
        )

    def forward(self, x):
        for layer in self.layers:
            x = x @ layer
        return x

@torch.inference_mode()
def benchmark():
    device = torch.device("cuda:0")

    x = torch.randn(8192, 8192, dtype=DTYPE, device=device)

    # Xavier 初始化，防止多层叠乘数值爆炸
    scale = 1.0 / math.sqrt(8192)
    pinned_cpu_weights = [
        (torch.randn(8192, 8192, dtype=DTYPE) * scale).pin_memory()
        for _ in range(NUM_LAYERS)
    ]

    model_base = BaseMLPModel(NUM_LAYERS, pinned_cpu_weights, device)

    # warmup
    for _ in range(2):
        out_std = model_base(x)
   
    iters = 5
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # 1. base model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    start.record()
    for _ in range(iters):
        out_std = model_base(x)
    end.record()
    torch.cuda.synchronize()
    std_time = start.elapsed_time(end) / iters
    std_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
```

## 3. 双 buffer streaming onload layer 推理

现代 GPU 内部拥有独立的计算引擎和双向拷贝引擎（Copy Engine），因此天然支持 host->device、compute、device->host 三种不同操作在物理层面上的异步并行。对于同类型的操作，受限于 PCIe 物理带宽，同向的数据拷贝通常只能排队串行。而只要显卡的 SM（流多处理器）资源未被占满，不同 Stream 中的多个计算（Compute）任务是可以并发执行的。

那么显而易见，在我们这个场景，将任务划分为两条流水线，以实现‘边搬运权重边计算’的交替作业。在时间序列下，可以表示为下图的样子：

```yaml
时间轴：t0 ----> t1 ----> t2 ----> t3 ----> t4 ---->
Copy 流：[Layer0] [Layer1] [Layer2] [Layer3] ...
计算流：          [Layer0] [Layer1] [Layer2] ...
```

时间开销：做到了不同 layer 之间的权重 copy 和计算的重叠。单步的真实耗时从串行的 (copy + compute) 变为了 max(copy, compute)，而减少的 min(copy, compute) 时间开销，就是被流水线隐藏的时延。
空间开销：特别地，要做到 Copy 和计算重叠，显然它们操作的就不能是同一块显存 buffer。因此两条流水线需要双 buffer，N 条流水线需要 N buffer（多 buffer 还有个响亮的名字叫 Ring Buffer）。

更具体一点，利用 pytorch 提供的 CUDA stream，我们可以实现双 buffer 双流水线逐层计算，流程如下：

- prologue： 在 copy stream 上先 load 第 0 层权重到 buffer_read，并记录「拷贝完成」事件。
- main loop：
  - copy stream
    - 当层数大于 0 时，等待 compute stream 上一次的「计算完成」事件。
    - 继续在 copy stream 上加载下一层权重到 buffer_write，并记录「拷贝完成」事件
  - compute stream
    - 等待 copy stream 上一次的「拷贝完成」事件，就地进行计算，计算完记录「计算完成」事件
  - 交换读写 buffer 的索引
- epilogue：将最终计算结果 copy 回 Host。
得益于 PyTorch 的流管理机制，本层的输出在计算流（compute stream）上完成，外部调用若使用默认流可安全消费。但在复杂拓扑中，务必注意跨流张量的显式同步。

代码如下：

```python
class DualStreamModel(nn.Module):
    """double buffer, double stream"""

    def __init__(self, num_layers, cpu_weights, device):
        super().__init__()
        self.num_layers = num_layers
        self.device = device

        self.cpu_weights = cpu_weights

        self.weight_buffers = [
            torch.empty((FEAT_DIM, FEAT_DIM), dtype=DTYPE, device=self.device),
            torch.empty((FEAT_DIM, FEAT_DIM), dtype=DTYPE, device=self.device),
        ]

        # copy stream, two events
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.events_copy_done = [torch.cuda.Event(), torch.cuda.Event()]
        self.event_compute_done = [torch.cuda.Event(), torch.cuda.Event()]

    def forward(self, x):
        compute_stream = torch.cuda.current_stream(self.device)

        # --- prologue ---
        with torch.cuda.stream(self.copy_stream):
            self.weight_buffers[0].copy_(self.cpu_weights[0], non_blocking=True)
            self.events_copy_done[0].record(self.copy_stream)
        read_idx, write_idx = 0, 1

        # --- main loop ---
        for i in range(self.num_layers):

            # 1. 【后台】异步预取下一层 i+1 到 write_idx buffer
            if i + 1 < self.num_layers:
                if i >= 1:
                    self.copy_stream.wait_event(self.event_compute_done[write_idx])

                with torch.cuda.stream(self.copy_stream):
                    self.weight_buffers[write_idx].copy_(
                        self.cpu_weights[i + 1], non_blocking=True
                    )
                    self.events_copy_done[write_idx].record(self.copy_stream)

            # 2. 【前台】等待当前 buffer copy 完毕并就地计算
            compute_stream.wait_event(self.events_copy_done[read_idx])
            x = x @ self.weight_buffers[read_idx]
            self.event_compute_done[read_idx].record(compute_stream)

            # 交换 buffer
            read_idx ^= 1
            write_idx ^= 1

        return x

@torch.inference_mode()
def benchmark():
    device = torch.device("cuda:0")

    x = torch.randn(8192, 8192, dtype=DTYPE, device=device)

    # Xavier 初始化，防止多层叠乘数值爆炸
    scale = 1.0 / math.sqrt(8192)
    # 注意：Host 端的 Tensor 必须使用 pin_memory() 锁页，配合 non_blocking=True 才能真正让 Copy 操作在后台流中异步执行，否则会退化为阻塞的同步传输
    pinned_cpu_weights = [
        (torch.randn(8192, 8192, dtype=DTYPE) * scale).pin_memory()
        for _ in range(NUM_LAYERS)
    ]

    model_base = BaseMLPModel(NUM_LAYERS, pinned_cpu_weights, device)
    model_stream = DualStreamModel(NUM_LAYERS, pinned_cpu_weights, device)

    # warmup
    for _ in range(2):
        out_std = model_base(x)
        out_stream = model_stream(x)
    torch.cuda.synchronize()

    iters = 5
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # 1. base model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    start.record()
    for _ in range(iters):
        out_std = model_base(x)
    end.record()
    torch.cuda.synchronize()
    std_time = start.elapsed_time(end) / iters
    std_mem = torch.cuda.max_memory_allocated(device) / (1024**2)

    # 2. double stream model
    del model_base
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    start.record()
    for _ in range(iters):
        out_stream = model_stream(x)
    end.record()
    torch.cuda.synchronize()
    off_time = start.elapsed_time(end) / iters
    off_mem = torch.cuda.max_memory_allocated(device) / (1024**2)

    assert torch.allclose(out_std, out_stream, rtol=1e-2, atol=1e-2), "diff error"

    # 3. 核心指标对比
    speed_retention = (std_time / off_time) * 100 if off_time > 0 else 0
    mem_saved_mb = std_mem - off_mem
    mem_reduction_pct = (mem_saved_mb / std_mem) * 100 if std_mem > 0 else 0

    print(f"\n=== {NUM_LAYERS} 层网络 BF16 极限压测报告 (Batch: {MAX_BATCH_SIZE}) ===")
    print("-" * 88)
    print(
        f"{'评估维度':<20} | {'BaseModel (全量驻留)':<14} | {'Streaming Onload (双缓冲)':<20} | {'对比差值 / 收益'}"
    )
    print("-" * 88)
    print(
        f"{'单次推理耗时 (ms)':<16} | {std_time:>20.2f} | {off_time:>14.2f} | 性能保持率: {speed_retention:.2f}%"
    )
    print(
        f"{'峰值显存占用 (MB)':<16} | {std_mem:>20.2f} | {off_mem:>14.2f} | 降低占比: {mem_reduction_pct:.2f}%"
    )
    print("-" * 88)
    print(
        f" -> 绝对收益: 仅牺牲了 {100 - speed_retention:.2f}% 的计算耗时，换取了 {mem_saved_mb:.2f} MB 的物理显存空间。"
    )
```

## 4. Benchmark 输出

测试环境为 RTX 5060 Laptop GPU，因为算力特别低，我特意选取的 data shape 使得计算开销几乎完美隐藏了 copy 开销（教学演示用途）：

```yaml
=== 24 层网络 BF16 极限压测报告 (Batch: 8192) ===
----------------------------------------------------------------------------------------
评估维度               | BaseModel (全量驻留) | Streaming Onload (双缓冲) | 对比差值 / 收益
----------------------------------------------------------------------------------------
单次推理耗时 (ms)      |        1101.41 |        1108.22 | 性能保持率：99.39%
峰值显存占用 (MB)      |        3976.12 |         904.12 | 降低占比：77.26%
----------------------------------------------------------------------------------------
 -> 绝对收益：仅牺牲了 0.61% 的计算耗时，换取了 3072.00 MB 的物理显存空间。
```

## 5. 总结

- 我们使用 pytorch CUDA stream 完成了一个 double buffer + double stream 的 计算 和 copy overlap 前向推理
  - 在几乎不损失推理性能的前提下，减少了 75% 空间占用。
- 这个 case 是我专门凑出来，用于演示多级流水线的空间复杂度优化能力的。在真实业务场景中，流水线本身更是优化时延开销的利器（利用了硬件并行的天然优势），但具体场景需要具体分析。
  - 特别地，此 case 没有涉及 offloading。如大语言模型推理中对产生的 KV Cache 进行 Streaming Offload/Onload 取用是很常见的优化，但核心思路与本文如出一辙。

我记得之前部署 Wan 2.2 I2V Pipeline 时，它会按高低噪声时间步切换不同的 DiT 模型。由于两个模型权重加视频激活值极其巨大，双卡 H20 都没法同时放下两个模型，所以官方 Demo 也是给的 FSDP 推理实现。这也是大模型时代显存受限场景的真实缩影。

希望这个极简的 Case 能给大家一点系统优化的直觉与收获。

如有错误，欢迎指正，感谢阅读！

以上

完整代码和测试脚本见 github [vitamin-cuda](https://github.com/WingEdge777/vitamin-cuda/tree/main/infer/torch/overlap_infer)
