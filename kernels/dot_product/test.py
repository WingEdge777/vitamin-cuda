import time
from functools import partial
from typing import Optional

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

common_flags = ["-O3", "-std=c++17"]
# Load the CUDA kernel as a python module
lib = load(
    name="dot_product_lib",
    sources=["dot_product.cu"],
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


def benchmark(op, a, b, warmup=10, rep=1000, prefix="torch"):
    # warm up
    for i in range(warmup):
        res = op(a, b)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(rep):
        res = op(a, b)
    torch.cuda.synchronize()
    print(f"{prefix:30s} mean time: {(time.perf_counter() - start) / rep * 1000:.6f} ms")
    return res


def diff_check(a, b, prefix="torch", rel=1e-4, abl=1e-4):
    message = f"{prefix} result diff"
    assert (
        torch.max(torch.abs(a - b)).item() < abl
        or torch.max(torch.abs(a - b) / torch.abs(a)).item() < rel
    ), message


if __name__ == "__main__":
    # test the kernel
    device = torch.device("cuda")
    sz = [1024, 2048, 4096]
    for n in sz:
        for m in sz:
            print("#" * 100)
            print(f"n: {n}, m: {m}")
            a = torch.randn(n * m).float().cuda()
            b = torch.randn(n * m).float().cuda()

            c = benchmark(torch.dot, a, b)
            c_my = benchmark(lib.dot_product, a, b, prefix="dot_product")
            # print(c, c_my)
            diff_check(c, c_my, prefix="dot_product")
            c_my = benchmark(
                lib.dot_product_fp32x4,
                a,
                b,
                prefix="dot_product_fp32x4",
            )
            diff_check(c, c_my, prefix="dot_product_fp32x4")

            ################### half
            a = a.half()
            b = b.half()
            c = benchmark(torch.dot, a, b)
            c_my = benchmark(lib.dot_product, a, b, prefix="dot_product")
            # print(c, c_my)
            diff_check(c, c_my, prefix="dot_product", abl=1e-2, rel=1e-2)
            c_my = benchmark(
                lib.dot_product_fp16x2,
                a,
                b,
                prefix="dot_product_fp16x2",
            )
            diff_check(c, c_my, prefix="dot_product_fp16x2", abl=1e-2, rel=1e-2)
            c_my = benchmark(
                lib.dot_product_fp16x8_packed,
                a,
                b,
                prefix="dot_product_fp16x8_packed",
            )
            diff_check(c, c_my, prefix="dot_product_fp16x8_packed", abl=1e-2, rel=1e-2)
