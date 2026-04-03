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
    extra_ldflags=["-L/usr/local/cuda-12.9/lib64/stubs", "-lcuda"],
    verbose=True,
)

baseline = None


def benchmark(op, a, b, c=None, warmup=10, rep=200, prefix="torch"):
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

    tflops = a.shape[0] * b.shape[0] * b.shape[1] * 2 * rep / 1e12 / duration
    if prefix == "torch":
        global baseline
        baseline = duration
        print(
            f"{prefix:40s} mean time: {duration / rep * 1000:8.6f} ms, {tflops:.2f} tflops"
        )
    else:
        speedup = baseline / duration
        print(
            f"{prefix:40s} mean time: {duration / rep * 1000:8.6f} ms, speedup: {speedup:.2f}, tflops: {tflops:.2f}"
        )


def diff_check(a, b, prefix="torch", eps=0.016):
    if not torch.allclose(a, b, atol=eps, rtol=eps):
        diff = torch.abs(a - b)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        print(f"{prefix} result mean diff: {mean_diff:.6f}, max diff: {max_diff:.6f}")
    assert torch.allclose(a, b, atol=eps, rtol=eps), "result diff"


def test_all():
    # test the kernel
    ns = [256, 512, 1024, 4096, 8192]
    torch.manual_seed(42)
    # for type in [torch.bfloat16, torch.float16]:
    for type in [torch.bfloat16]:
        for n in ns:
            print("#" * 100)
            print(f"m: {n}, n: {n}, k: {n}")
            # a x b = c
            a = torch.randn(n, n).to(type).cuda()
            b = torch.randn(n, n).to(type).cuda()
            c = torch.zeros(n, n).to(type).cuda()
            print("#" * 100)
            # cuda core
            benchmark(partial(torch.matmul, out=c), a, b)
            c_cublas = torch.zeros_like(c)
            benchmark(
                lib.hgemm_cublas,
                a,
                b,
                c_cublas,
                prefix="hgemm_cublas",
            )
            diff_check(c, c_cublas, prefix="hgemm_cublas")


def test_4096():

    m, n, k = [4096] * 3
    print("#" * 100)
    print(f"n: {n}, m: {m}, k: {k}")
    # for type in [torch.bfloat16, torch.float16]:
    for type in [torch.bfloat16]:
        # a x b = c
        a = torch.randn(m, k).to(type).cuda()
        b = torch.randn(k, n).to(type).cuda()
        c = torch.zeros(m, n).to(type).cuda()

        benchmark(partial(torch.matmul, out=c), a, b)

        c_cublas = torch.zeros_like(c)
        benchmark(
            lib.hgemm_cublas,
            a,
            b,
            c_cublas,
            prefix="hgemm_cublas",
        )
        diff_check(c, c_cublas, prefix="hgemm_cublas")

        c_my = torch.zeros_like(c)

        benchmark(lib.hgemm_bcf_dbf_rw, a, b, c_my, prefix="hgemm_bcf_dbf_rw")
        diff_check(c, c_my, prefix="hgemm_bcf_dbf_rw")

        benchmark(lib.hgemm_k_stages, a, b, c_my, prefix="hgemm_k_stages")
        diff_check(c, c_my, prefix="hgemm_k_stages")

        benchmark(lib.hgemm_tma_r_k_stages_64, a, b, c_my, prefix="hgemm_tma_r_k_stages_64")
        diff_check(c, c_my, prefix="hgemm_tma_r_k_stages_64")

        benchmark(lib.hgemm_tma_r_k_stages_32, a, b, c_my, prefix="hgemm_tma_r_k_stages_32")
        diff_check(c, c_my, prefix="hgemm_tma_r_k_stages_32")


if __name__ == "__main__":
    # test_all()
    test_4096()
