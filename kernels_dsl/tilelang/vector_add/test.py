import sys
from functools import partial
import time
import torch
import tilelang
import tilelang.language as T

import triton
import triton.language as tl

sys.path.append("../../../")
from kernels.elementwise.test import lib


# triton
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    st = pid * BLOCK_SIZE
    offsets = st + tl.arange(0, BLOCK_SIZE)

    if st + BLOCK_SIZE <= n:
        x = tl.load(x_ptr + offsets)
        y = tl.load(y_ptr + offsets)
        out = x + y
        tl.store(out_ptr + offsets, out)
    else:
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        out = x + y
        tl.store(out_ptr + offsets, out, mask=mask)


# tilelang
@tilelang.jit
def add_tilelang(N: int, block: int = 256, dtype: str = "float16"):

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
            for i in T.Parallel(block):
                out_id = bx * block + i
                C[out_id] = A[out_id] + B[out_id]

    return main


@tilelang.jit
def add_tilelang_vectorized(N: int, block: int = 256, dtype: str = "float16"):
    vec = 8

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        grid = T.ceildiv(N, block * vec)

        with T.Kernel(grid, threads=block) as bx:
            tid = T.get_thread_binding(0)
            tile_id = bx * block + tid
            base = tile_id * vec

            if base + 7 < N:
                for i in T.vectorized(vec):
                    elem = base + i
                    C[elem] = A[elem] + B[elem]
            else:
                for i in T.serial(N - base):
                    elem = base + i
                    C[elem] = A[elem] + B[elem]

    return main


def diff_check(a, b, prefix="torch", eps=1e-3):
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


def test_all():
    torch.manual_seed(42)
    DEVICE = torch.device("cuda")
    for n in [1024, 4096, 1024 * 32, 1024 * 1024, 1024 * 4096, 4096 * 4096]:
        print("#" * 100)
        print(f"vector add, n: {n}")
        x = torch.randn(n, dtype=torch.float32, device=DEVICE)
        y = torch.randn(n, dtype=torch.float32, device=DEVICE)
        out = torch.empty_like(x)

        benchmark(partial(torch.add, out=out), x, y)

        out_my = torch.zeros_like(out)

        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        benchmark(
            partial(add_kernel[grid], n=n, BLOCK_SIZE=256),
            x,
            y,
            out_my,
            prefix="triton",
        )
        diff_check(out, out_my)
        kernel = add_tilelang(n, dtype="float32")
        benchmark(kernel, x, y, out_my, prefix="tilelang")
        benchmark(
            lib.elementwise_add_fp32x4,
            x,
            y,
            out_my,
            prefix="elementwise_add_fp32x4",
        )
        diff_check(out, out_my, prefix="elementwise_add_fp32x4")

        x = x.half()
        y = y.half()
        out = out.half()
        out_my = out_my.half()
        benchmark(partial(torch.add, out=out), x, y)
        benchmark(
            partial(add_kernel[grid], n=n, BLOCK_SIZE=256),
            x,
            y,
            out_my,
            prefix="triton",
        )
        kernel = add_tilelang(n, dtype="float16")
        print(kernel.get_kernel_source())
        benchmark(kernel, x, y, out_my, prefix="tilelang")
        diff_check(out, out_my)
        benchmark(
            lib.elementwise_add_fp16x8_packed,
            x,
            y,
            out_my,
            prefix="elementwise_add_fp16x8_packed",
        )
        diff_check(out, out_my, prefix="elementwise_add_fp16x8_packed")


def test_sp():
    torch.manual_seed(42)
    DEVICE = torch.device("cuda")
    for n in [4096 * 4096, 4096 * 4096 + 1]:
        print("#" * 100)
        print(f"vector add, n: {n}")
        x = torch.randn(n, dtype=torch.float32, device=DEVICE)
        y = torch.randn(n, dtype=torch.float32, device=DEVICE)
        out = torch.empty_like(x)

        out_my = torch.zeros_like(out)

        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

        x = x.half()
        y = y.half()
        out = out.half()
        out_my = out_my.half()
        benchmark(partial(torch.add, out=out), x, y)
        benchmark(
            partial(add_kernel[grid], n=n, BLOCK_SIZE=1024),
            x,
            y,
            out_my,
            prefix="triton",
        )
        kernel = add_tilelang(n, dtype="float16")
        # print(kernel.get_kernel_source())
        out_my = torch.zeros_like(out_my)
        benchmark(kernel, x, y, out_my, prefix="tilelang")

        kernel = add_tilelang_vectorized(n, dtype="float16")
        # print(kernel.get_kernel_source())
        out_my = torch.zeros_like(out_my)
        benchmark(kernel, x, y, out_my, prefix="tilelang_vectorized")
        # print(out[-1], out_my[-1])
        diff_check(out, out_my)
        benchmark(
            lib.elementwise_add_fp16x8_packed,
            x,
            y,
            out_my,
            prefix="elementwise_add_fp16x8_packed",
        )
        diff_check(out, out_my, prefix="elementwise_add_fp16x8_packed")


if __name__ == "__main__":
    # test_all()
    test_sp()
