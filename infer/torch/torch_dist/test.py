import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


def setup_dist(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=rank)
    torch.cuda.set_device(rank)


def cleanup_dist():
    dist.destroy_process_group()


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


@torch.inference_mode()
def benchmark(fn, *args, ref=None, warmup=3, repeat=10):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    start = time.perf_counter()
    for _ in range(repeat):
        out = fn(*args)
    torch.cuda.synchronize()
    avg_time = (time.perf_counter() - start) / repeat * 1000
    if ref is not None:
        print(
            f"Output match reference: {torch.allclose(out, ref.cuda(), rtol=1e-4, atol=1e-4)}, Avg time: {avg_time:.3f} ms"
        )
    return out, avg_time


def run_demo(rank, cls, label, world_size, model, x, ref):
    setup_dist(rank, world_size)
    dist_model = cls(model).cuda()
    device_x = x.cuda()
    if rank == 0:
        print(f"[{label}]", end=" ")
        benchmark(dist_model, device_x, ref=ref)
    else:
        benchmark(dist_model, device_x)
    cleanup_dist()


if __name__ == "__main__":
    world_size = 2
    hidden_size = 8192
    batch_size = 1024
    # 注意这里使用了较大的batch_size，为了体现DP/TP的加速作用，如果数据规模较小的话通信开销反而更大会导致DP/TP更慢，但在大模型里几乎不存在这个问题

    torch.manual_seed(42)
    model = TwoLayerModel(hidden_size)
    x = torch.randn(batch_size, hidden_size)
    with torch.inference_mode():
        ref = model(x)

    demos = [("Single GPU", None), ("DP", DPModel), ("TP", TPModel), ("PP", PPModel)]
    for i, (label, cls) in enumerate(demos):
        print(f"\n{'='*50}\nDemo {i}: {label}\n{'='*50}")
        if cls is None:
            model.cuda()
            x_cuda = x.cuda()
            print(f"[{label}]", end=" ")
            benchmark(model, x_cuda, ref=ref)
            model.cpu()
        else:
            mp.spawn(
                run_demo,
                args=(cls, label, world_size, model, x, ref),
                nprocs=world_size,
                join=True,
            )
