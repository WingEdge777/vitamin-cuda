import time
from functools import partial
from typing import Optional

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

common_flags = ["-O3", "-std=c++17"]
# Load the CUDA kernel as a python module
lib = load(
    name="hgemm_sm120_lib",
    sources=["hgemm_sm120.cu", "cublas.cu"],
    extra_cuda_cflags=common_flags
    + [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-Xptxas -v",
    ],
    extra_cflags=common_flags,
    verbose=True,
)

baseline = 1e-6


def benchmark(op, a, b, c=None, warmup=0, rep=1, prefix="torch"):
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


def run():
    m, n, k = [4096] * 3
    print("#" * 100)
    print(f"n: {n}, m: {m}, k: {k}")
    # a x b = c
    a = torch.randn(m, k).bfloat16().cuda()
    b = torch.randn(k, n).bfloat16().cuda()
    c = torch.zeros(m, n).bfloat16().cuda()


    c_cublas = torch.zeros_like(c)
    benchmark(
        lib.hgemm_cublas,
        a,
        b,
        c_cublas,
        prefix="hgemm_cublas",
    )

    c_my = torch.zeros_like(c)

    benchmark(lib.hgemm_naive, a, b, c_my, prefix="hgemm_naive")
    benchmark(lib.hgemm_bcf, a, b, c_my, prefix="hgemm_bcf")
    benchmark(lib.hgemm_bcf_dbf, a, b, c_my, prefix="hgemm_bcf_dbf")
    benchmark(lib.hgemm_bcf_dbf_rw, a, b, c_my, prefix="hgemm_bcf_dbf_rw")


if __name__ == "__main__":
    run()
