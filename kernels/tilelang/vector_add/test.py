from functools import partial
import time
import torch
import tilelang
import tilelang.language as T

import triton
import triton.language as tl

# triton
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

# triton
@tilelang.jit
def add_tilelang(N: int, block: int = 256, dtype: str = "float32"):

    @T.prim_func
    def add_kernel(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
            for i in T.Parallel(block):
                out_id = bx * block + i
                C[out_id] = A[out_id] + B[out_id]

    return add_kernel


def diff_check(a, b, prefix="torch", eps=1e-5):
    if not torch.allclose(a, b, atol=eps, rtol=eps):
        diff = torch.abs(a - b)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        print(f"{prefix} result mean diff: {mean_diff:.6f}, max diff: {max_diff:.6f}")
    assert torch.allclose(a, b, atol=eps, rtol=eps), "result diff"


def benchmark(op, x, y, o=None, warmup=10, rep=1000, prefix="torch"):
    if o is not None:
        # warm up
        for i in range(warmup):
            op(x, y, o)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(rep):
            op(x, y, o)
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
    DEVICE = torch.device("cuda")
    for n in [1024, 4096, 1024 * 32, 1024 * 1024, 1024*4096, 4096*4096]:
        print("#" * 100)
        print(f"vector add, n: {n}")
        x = torch.randn(n, dtype=torch.float32, device=DEVICE)
        y = torch.randn(n, dtype=torch.float32, device=DEVICE)
        out = torch.empty_like(x)

        benchmark(partial(torch.add, out=out), x, y)

        out_my = torch.zeros_like(out)

        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        benchmark(partial(add_kernel[grid], n=n, BLOCK_SIZE=256), x, y, out_my, prefix="triton")
        diff_check(out, out_my)

        kernel = add_tilelang(n)
        benchmark(kernel, x, y, out_my, prefix="tilelang")
        diff_check(out, out_my)

# tvm_target = determine_target("cuda -arch=sm_120", return_object=True)
# kernel = tilelang.compile(func, target=tvm_target)
