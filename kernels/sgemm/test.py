import time
from functools import partial
from typing import Optional

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

common_flags = ["-O3", "-std=c++17"]
# Load the CUDA kernel as a python module
lib = load(
    name="gemm_lib",
    sources=["sgemm.cu", "cublas.cu"],
    extra_cuda_cflags=common_flags
    + [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=common_flags,
    verbose=True,
)

baseline = None


def benchmark(op, a, b, c=None, warmup=10, rep=100, prefix="torch"):
    if c is not None:
        # warm up
        for i in range(warmup):
            op(a, b, c)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(rep):
            op(a, b, c)
        torch.cuda.synchronize()
    else:
        # warm up
        for i in range(warmup):
            op(a, b)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(rep):
            op(a, b)
        torch.cuda.synchronize()

    duration = time.perf_counter() - start

    if prefix == "torch":
        global baseline
        baseline = duration
        print(f"{prefix:30s} mean time: {duration / rep * 1000:.6f} ms")
    else:
        speedup = baseline / duration
        print(
            f"{prefix:30s} mean time: {duration / rep * 1000:.6f} ms, speedup: {speedup:.2f}"
        )


def diff_check(a, b, prefix="torch", eps=1e-3):
    if "tf32" in prefix:
        eps = 0.1
    if not torch.allclose(a, b, atol=eps, rtol=eps):
        print(f"{prefix} result diff: {torch.mean(torch.abs(a - b)).item()}")
    assert torch.allclose(a, b, atol=eps, rtol=eps), "result diff"


if __name__ == "__main__":
    # test the kernel
    nmk = [512, 1024, 2048, 4096, 8192]
    torch.manual_seed(42)
    for n in nmk:
        for m in nmk:
            for k in nmk:
                print("#" * 100)
                print(f"n: {n}, m: {m}")
                # a dot b = c
                a = torch.randn(n, m).float().cuda()
                b = torch.randn(m, k).float().cuda()
                c = torch.zeros(n, k).float().cuda()

                benchmark(partial(torch.matmul, out=c), a, b)
                c_cublas = torch.zeros_like(c)

                benchmark(lib.sgemm_cublas, a, b, c_cublas, prefix="sgemm_cublas")
                diff_check(c, c_cublas, prefix="sgemm_cublas")
                benchmark(
                    lib.sgemm_cublas_tf32,
                    a,
                    b,
                    c_cublas,
                    prefix="sgemm_cublas_tf32",
                )
                diff_check(c, c_cublas, prefix="sgemm_cublas_tf32")
