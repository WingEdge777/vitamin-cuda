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


if __name__ == "__main__":
    benchmark()
