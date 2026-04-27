import time

import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    st = pid * BLOCK_SIZE
    offsets = st + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    out = x + y

    tl.store(out_ptr + offsets, out, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    out = torch.empty_like(x)
    n = out.numel()

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    add_kernel[grid](x, y, out, n, BLOCK_SIZE=256)

    return out


def diff_check(a, b, prefix="torch", eps=1e-5):
    if not torch.allclose(a, b, atol=eps, rtol=eps):
        diff = torch.abs(a - b)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        print(f"{prefix} result mean diff: {mean_diff:.6f}, max diff: {max_diff:.6f}")
    assert torch.allclose(a, b, atol=eps, rtol=eps), "result diff"


def benchmark(op, x, y, out=None, warmup=10, rep=100, prefix="torch"):
    if out is not None:
        # warm up
        for i in range(warmup):
            op(x, y, out)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(rep):
            op(x, y, out)
        torch.cuda.synchronize()
    else:
        # warm up
        for i in range(warmup):
            o = op(x, y)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(rep):
            o = op(x, y)
        torch.cuda.synchronize()

    duration = time.perf_counter() - start

    if prefix == "torch":
        global baseline
        baseline = duration
        print(f"{prefix:40s} mean time: {duration / rep * 1000:8.6f} ms")
    else:
        speedup = baseline / duration
        print(
            f"{prefix:40s} mean time: {duration / rep * 1000:8.6f} ms, speedup: {speedup:.2f}"
        )
    return o


if __name__ == "__main__":
    torch.manual_seed(42)
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    for n in [128, 512, 1024, 4096, 10240, 1024 * 1024]:
        print("#" * 100)
        print(f"vector add, n: {n}")
        x = torch.randn(n, dtype=torch.float32, device=DEVICE)
        y = torch.randn(n, dtype=torch.float32, device=DEVICE)

        out = benchmark(torch.add, x, y)

        out_triton = benchmark(add, x, y, prefix="triton")

        diff_check(out, out_triton)
