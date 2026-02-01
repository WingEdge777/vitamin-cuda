import time
from functools import partial
from typing import Optional

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

common_flags = ["-O3", "-std=c++17"]
# Load the CUDA kernel as a python module
lib = load(
    name="reduce_sum_lib",
    sources=["reduce_sum.cu"],
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


def benchmark(op, a, warmup=10, rep=1000, prefix="torch"):
    # warm up
    for i in range(warmup):
        res = op(a)
    torch.cuda.synchronize()
    start = time.time()
    for i in range(rep):
        res = op(a)
    torch.cuda.synchronize()
    print(
        f"{prefix:30s} mean time: {(time.time() - start) / rep * 1000:.6f} ms"
    )
    return res


def diff_check(a, b, prefix="torch", rel=1e-5, abl = 1e-5):
    message = f"{prefix} result diff"
    assert torch.max(torch.abs(a - b)).item() < abl or torch.max(torch.abs(a - b) / torch.abs(a)).item() < rel, message


if __name__ == "__main__":
    # test the kernel
    device = torch.device("cuda")
    sz = [1024, 2048, 4096]
    # sz = [2, 2, 2]
    torch.manual_seed(42)
    for n in sz:
        for m in sz:
            print("#" * 100)
            print(f"n: {n}, m: {m}")
            a = torch.randn(n, m).float().cuda()
            out = benchmark(torch.sum, a)
            out_my = benchmark(lib.reduce_sum, a, prefix="reduce_sum")
            diff_check(out, out_my, prefix="reduce_sum")
            out_my = benchmark(lib.reduce_sum_fp32x4, a, prefix="reduce_sum_fp32x4")
            diff_check(out, out_my, prefix="reduce_sum_fp32x4")

            ################### half
            a = a.half()
            out = benchmark(torch.sum, a)
            out_my = benchmark(lib.reduce_sum, a, prefix="reduce_sum_half")
            diff_check(out, out_my, prefix="reduce_sum_half", rel=1e-3, abl=1e-3)
            out_my = benchmark(lib.reduce_sum_fp16x2, a, prefix="reduce_sum_fp16x2")
            diff_check(out, out_my, prefix="reduce_sum_fp16x2")
            out_my = benchmark(lib.reduce_sum_fp16x8_packed, a, prefix="reduce_sum_fp16x8_packed")
            diff_check(out, out_my, prefix="reduce_sum_fp16x8_packed")

            ################## int8
            a = torch.randint(-128, 127, (n, m) , dtype=torch.int8, device="cuda")
            out = benchmark(torch.sum, a)
            out_my = benchmark(lib.reduce_sum_i8, a, prefix="reduce_sum_i8")
            diff_check(out, out_my, prefix="reduce_sum_i8")
            out_my = benchmark(lib.reduce_sum_i8x16_packed, a, prefix="reduce_sum_i8x16_packed")
            diff_check(out, out_my, prefix="reduce_sum_i8x16_packed")
            out_my = benchmark(lib.reduce_sum_i8x16_packed_dp4a, a, prefix="reduce_sum_i8x16_packed_dp4a")
            diff_check(out, out_my, prefix="reduce_sum_i8x16_packed_dp4a")
