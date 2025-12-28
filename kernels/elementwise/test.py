import time
from functools import partial
from typing import Optional

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

common_flags = ["-O3", "-std=c++17"]
# Load the CUDA kernel as a python module
lib = load(
    name="elementwise_lib",
    sources=["elementwise.cu"],
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


def benchmark(op, a, b, c=None, warmup=10, rep=500, prefix="torch"):
    if c is not None:
        # warm up
        for i in range(10):
            op(a, b, c)
        torch.cuda.synchronize()
        start = time.time()
        for i in range(rep):
            op(a, b, c)
        torch.cuda.synchronize()
        print(f"{prefix:30s} mean time: {(time.time() - start) / rep * 1000:.6f} ms")
    else:
        # warm up
        for i in range(10):
            op(a, b)
        torch.cuda.synchronize()
        start = time.time()
        for i in range(rep):
            op(a, b)
        torch.cuda.synchronize()
        print(f"{prefix:30s} mean time: {(time.time() - start) / rep * 1000:.6f} ms")


def diff_check(a, b, prefix="torch"):
    assert torch.max(torch.abs(a - b)).item() < 1e-5, f"{prefix} result diff"


if __name__ == "__main__":
    # test the kernel
    device = torch.device("cuda")
    sz = [1024, 2048, 4096]
    for n in sz:
        for m in sz:
            print("#" * 100)
            print(f"n: {n}, m: {m}")
            a = torch.randn(n, m).float().cuda()
            b = torch.randn(n, m).float().cuda()
            c = torch.empty_like(a).float().cuda()

            benchmark(partial(torch.add, out=c), a, b)
            c_my = torch.empty_like(a)
            benchmark(lib.elementwise_add, a, b, c_my, prefix="elementwise_add")
            diff_check(c, c_my, prefix="elementwise_add")
            benchmark(lib.elementwise_add_fp32x4, a, b, c_my, prefix="elementwise_add_fp32x4")
            diff_check(c, c_my, prefix="elementwise_add_fp32x4")

            ################### half
            a = a.half()
            b = b.half()
            c = c.half()
            benchmark(partial(torch.add, out=c), a, b)
            c_my = c_my.half()
            benchmark(lib.elementwise_add, a, b, c_my, prefix="ele_add_half")
            diff_check(c, c_my, prefix="ele_add_half")
            benchmark(
                lib.elementwise_add_fp16x2, a, b, c_my, prefix="elementwise_add_fp16x2"
            )
            diff_check(c, c_my, prefix="elementwise_add_fp16x2")
            benchmark(
                lib.elementwise_add_fp16x8, a, b, c_my, prefix="elementwise_add_fp16x8"
            )
            diff_check(c, c_my, prefix="elementwise_add_fp16x8")
            benchmark(
                lib.elementwise_add_fp16x8_packed,
                a,
                b,
                c_my,
                prefix="elementwise_add_fp16x8_packed",
            )
            diff_check(c, c_my, prefix="elementwise_add_fp16x8_packed")
