import time
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

common_flags = ["-O3", "-std=c++17"]
# Load the CUDA kernel as a python module
lib = load(
    name="relu6_lib",
    sources=["relu6.cu"],
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


def benchmark(op, a, b=None, warmup=10, rep=500, prefix="torch"):
    if b is not None:
        # warm up
        for i in range(warmup):
            op(a, b)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(rep):
            op(a, b)
        torch.cuda.synchronize()
        print(
            f"{prefix:30s} mean time: {(time.perf_counter() - start) / rep * 1000:.6f} ms"
        )
    else:
        # warm up
        for i in range(warmup):
            op(a)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(rep):
            op(a)
        torch.cuda.synchronize()
        print(
            f"{prefix:30s} mean time: {(time.perf_counter() - start) / rep * 1000:.6f} ms"
        )


def diff_check(a, b, prefix="torch", eps=1e-5):
    message = f"{prefix} result diff"
    assert torch.max(torch.abs(a - b)).item() < eps, message


if __name__ == "__main__":
    # test the kernel
    device = torch.device("cuda")
    sz = [1024, 2048, 4096]
    for n in sz:
        for m in sz:
            print("#" * 100)
            print(f"n: {n}, m: {m}")
            a = torch.randn(n, m).float().cuda()
            b = a.clone()

            benchmark(partial(F.relu6, inplace=True), b)
            b_my = torch.empty_like(a)

            benchmark(lib.relu6, a, b_my, prefix="relu6")
            b = F.relu6(a)
            diff_check(b, b_my, prefix="relu6")
            benchmark(lib.relu6_fp32x4, a, b_my, prefix="relu6_fp32x4")
            b = F.relu6(a)
            diff_check(b, b_my, prefix="relu6_fp32x4")

            ################### half
            a = a.half()
            b = a.clone()
            benchmark(partial(F.relu6, inplace=True), b)
            b_my = b_my.half()

            benchmark(lib.relu6, a, b_my, prefix="relu6_half")
            b = F.relu6(a)
            diff_check(b, b_my, prefix="relu6_half", eps=5e-3)
            benchmark(lib.relu6_fp16x2, a, b_my, prefix="relu6_fp16x2")
            b = F.relu6(a)
            diff_check(b, b_my, prefix="relu6_fp16x2", eps=5e-3)
            benchmark(lib.relu6_fp16x8, a, b_my, prefix="relu6_fp16x8")
            b = F.relu6(a)
            diff_check(b, b_my, prefix="relu6_fp16x8", eps=5e-3)
            benchmark(
                lib.relu6_fp16x8_packed,
                a,
                b_my,
                prefix="relu6_fp16x8_packed",
            )
            b = F.relu6(a)
            diff_check(b, b_my, prefix="relu6_fp16x8_packed", eps=1e-3)
