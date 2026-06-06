import sys
from functools import partial
import torch
import tilelang
import tilelang.language as T

import triton
import triton.language as tl
from flashinfer.testing.utils import bench_gpu_time
import numpy as np

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


@tilelang.jit
def add_tilelang_vectorized_eager(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dtype = "float16"
):
    N = T.const("N")
    block = 256
    vec = 128 // tilelang.DataType(dtype).bits
    A: T.Tensor((N,), dtype)
    B: T.Tensor((N,), dtype)
    C: T.Tensor((N,), dtype)

    with T.Kernel(T.ceildiv(N, block * vec), threads=block) as bx:
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

@tilelang.jit
def elementwise_add(A, B, C, block_N=2048, dtype="float16", threads=256):
    N = T.const("N")

    A: T.Tensor((N), dtype)
    B: T.Tensor((N), dtype)
    C: T.Tensor((N,), dtype)

    with T.Kernel(T.ceildiv(N, block_N), threads=threads) as bx:
        A_shared = T.alloc_shared((block_N), dtype)
        B_shared = T.alloc_shared((block_N), dtype)
        C_local = T.alloc_fragment((block_N), dtype)
        C_shared = T.alloc_shared((block_N), dtype)

        T.copy(A[bx * block_N], A_shared)
        T.copy(B[bx * block_N], B_shared)
        for local_x in T.Parallel(block_N):
            C_local[local_x] = A_shared[local_x] + B_shared[local_x]
        T.copy(C_local, C_shared)
        T.copy(C_shared, C[bx * block_N])

    # return C

@tilelang.jit
def elementwise_add_no_shared(A, B, C, block_N=2048, dtype="float16", threads=256):
    N = T.const("N")

    A: T.Tensor((N), dtype)
    B: T.Tensor((N), dtype)
    C: T.Tensor((N,), dtype)

    with T.Kernel(T.ceildiv(N, block_N), threads=threads) as bx:
        A_shared = T.alloc_fragment((block_N), dtype)
        B_shared = T.alloc_fragment((block_N), dtype)
        C_local = T.alloc_fragment((block_N), dtype)

        T.copy(A[bx * block_N], A_shared)
        T.copy(B[bx * block_N], B_shared)
        for local_x in T.Parallel(block_N):
            C_local[local_x] = A_shared[local_x] + B_shared[local_x]
        T.copy(C_local, C[bx * block_N])

def diff_check(a, b, prefix="torch", eps=1e-3):
    if not torch.allclose(a, b, atol=eps, rtol=eps):
        diff = torch.abs(a - b)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        print(f"{prefix} result mean diff: {mean_diff:.6f}, max diff: {max_diff:.6f}")
    assert torch.allclose(a, b, atol=eps, rtol=eps), "result diff"


def benchmark(op, x, y, o=None, prefix="torch"):
    def measure(fn, x, y, o, cold):
        times = bench_gpu_time(
            fn=fn,
            input_args=(x, y, o) if o is not None else (x, y),
            dry_run_iters=10,
            repeat_iters=100,
            enable_cupti=False,
            use_cuda_graph=False,  # pure CUDA-event timer
            cold_l2_cache=cold,
        )
        return float(np.median(times))

    avg_duration = measure(op, x, y, o, cold=True)
    if prefix == "torch":
        global baseline
        baseline = avg_duration
        print(f"{prefix:40s} mean time: {avg_duration:8.6f} ms")
    else:
        speedup = baseline / avg_duration
        print(
            f"{prefix:40s} mean time: {avg_duration:8.6f} ms, speedup: {speedup:.2f}"
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
        # print(kernel.get_kernel_source())
        out_my = torch.zeros_like(out_my)
        benchmark(kernel, x, y, out_my, prefix="tilelang")

        kernel = add_tilelang_vectorized(n, dtype="float16")
        # print(kernel.get_kernel_source())
        out_my = torch.zeros_like(out_my)
        benchmark(kernel, x, y, out_my, prefix="tilelang_vectorized")
        # print(out[-1], out_my[-1])
        diff_check(out, out_my)
        out_my = torch.zeros_like(out_my)
        benchmark(
            add_tilelang_vectorized_eager,
            x,
            y,
            out_my,
            prefix="tilelang_vectorized_eager",
        )
        # print(add_tilelang_vectorized_eager.compile(N=n, A=x).get_kernel_source())
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
        out_my = torch.zeros_like(out_my)
        benchmark(
            add_tilelang_vectorized_eager,
            x,
            y,
            out_my,
            prefix="tilelang_vectorized_eager",
        )
        # print(add_tilelang_vectorized_eager.compile(N=n, A=x).get_kernel_source())
        # print(out[-1], out_my[-1])
        out_my = torch.zeros_like(out_my)
        benchmark(
            elementwise_add,
            x,
            y,
            out_my,
            prefix="elementwise_add",
        )
        print(elementwise_add.compile(N=n).get_kernel_source())
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
