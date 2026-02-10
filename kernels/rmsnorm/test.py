import time
from functools import partial
from typing import Optional

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

common_flags = ["-O3", "-std=c++17"]
# Load the CUDA kernel as a python module
lib = load(
    name="rmsnorm_lib",
    sources=["rmsnorm.cu"],
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


def rmsnorm(x, weight, dim=1, eps=1e-6, out=None):  # [n, dim], [dim]
    out = x * torch.rsqrt(x.pow(2).mean(dim, keepdim=True) + eps) * weight
    return out


def benchmark(op, a, b, warmup=10, rep=300, prefix="torch"):
    # warm up
    for i in range(warmup):
        res = op(a, b)
    torch.cuda.synchronize()
    start = time.time()
    for i in range(rep):
        res = op(a, b)
    torch.cuda.synchronize()
    print(f"{prefix:30s} mean time: {(time.time() - start) / rep * 1000:.6f} ms")
    return res


def diff_check(a, b, prefix="torch", eps=1e-4):
    message = f"{prefix} result diff"
    assert torch.mean(torch.abs(a - b)).item() < eps, message


if __name__ == "__main__":
    # test the kernel
    device = torch.device("cuda")
    bs = [256, 1024, 2048, 4096]
    sz = [2048, 4096, 8192]
    torch.manual_seed(42)
    for n in bs:
        for m in sz:
            print("#" * 100)
            print(f"n: {n}, m: {m}")
            a = torch.randn(n, m).float().cuda()
            b = torch.randn(m).float().cuda()

            c = benchmark(rmsnorm, a, b)
            c_my = benchmark(lib.rmsnorm, a, b, prefix="rmsnorm")
            # print(b, b_my)
            diff_check(c, c_my, prefix="rmsnorm")

            benchmark(lib.rmsnorm_fp32x4, a, b, prefix="rmsnorm_fp32x4")
            # print(b, b_my)
            diff_check(c, c_my, prefix="rmsnorm_fp32x4")

            a = a.half()
            b = b.half()
            c = benchmark(rmsnorm, a, b)
            c_my = benchmark(lib.rmsnorm, a, b, prefix="rmsnorm_fp16")
            diff_check(c, c_my, prefix="rmsnorm_fp16", eps=1e-3)

            c_my = benchmark(
                lib.rmsnorm_fp16x8_packed, a, b, prefix="rmsnorm_fp16x8_packed"
            )
            diff_check(c, c_my, prefix="rmsnorm_fp16x8_packed", eps=1e-3)
