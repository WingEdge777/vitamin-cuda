import time
from functools import partial
from typing import Optional

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

common_flags = ["-O3", "-std=c++17"]
# Load the CUDA kernel as a python module
lib = load(
    name="transpose_lib",
    sources=["transpose.cu"],
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


def transpose(a, b):
    return torch.transpose_copy(a, dim0=0, dim1=1, out=b)


def benchmark(op, a, b=None, warmup=10, rep=500, prefix="torch"):
    if b is not None:
        # warm up
        for i in range(warmup):
            res = op(a, b)
        torch.cuda.synchronize()
        start = time.time()
        for i in range(rep):
            res = op(a, b)
        torch.cuda.synchronize()
        print(f"{prefix:30s} mean time: {(time.time() - start) / rep * 1000:.6f} ms")
    else:
        # warm up
        for i in range(warmup):
            res = op(a)
        torch.cuda.synchronize()
        start = time.time()
        for i in range(rep):
            res = op(a)
        torch.cuda.synchronize()
        print(f"{prefix:30s} mean time: {(time.time() - start) / rep * 1000:.6f} ms")
    return res


def diff_check(a, b, prefix="torch", eps=1e-4):
    message = f"{prefix} result diff"
    assert torch.mean(torch.abs(a - b)).item() < eps, message


if __name__ == "__main__":
    # test the kernel
    device = torch.device("cuda")
    seq_lens = [512, 1024, 2048, 4096, 8192]
    head_dim = [128, 256]
    torch.manual_seed(42)
    for n in seq_lens:
        for m in head_dim:
            print("#" * 100)
            print(f"n: {n}, m: {m}")
            a = torch.randn(n, m).float().cuda()
            b = torch.randn(m, n).float().cuda()

            benchmark(transpose, a, b)
            b_my = torch.empty_like(b)
            benchmark(lib.transpose, a, b_my, prefix="transpose")
            # print(b, b_my)
            diff_check(b, b_my, prefix="transpose")

            benchmark(lib.transpose_fp32x4, a, b_my, prefix="transpose_fp32x4")
            diff_check(b, b_my, prefix="transpose_fp32x4")
