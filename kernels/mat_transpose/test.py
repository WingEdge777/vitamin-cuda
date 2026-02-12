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

# @torch.compile()
def transpose(a, b):
    return torch.transpose_copy(a, dim0=0, dim1=1, out=b)

baseline = None

def benchmark(op, a, b=None, warmup=10, rep=300, prefix="torch"):
    # warm up
    for i in range(warmup):
        op(a, b)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(rep):
        op(a, b)
    torch.cuda.synchronize()
    if prefix == "torch":
        global baseline
        duration = time.perf_counter() - start
        baseline = duration
        print(f"{prefix:30s} mean time: {duration / rep * 1000:.6f} ms")
    else:
        duration = time.perf_counter() - start
        speedup = baseline / duration
        print(f"{prefix:30s} mean time: {duration / rep * 1000:.6f} ms, speedup: {speedup:.2f}")


def diff_check(a, b, prefix="torch", eps=1e-6):
    if not torch.allclose(a, b, atol=eps, rtol=eps):
        print(f"{prefix} result diff: {torch.mean(torch.abs(a - b)).item()}")
    assert torch.allclose(a, b, atol=eps, rtol=eps), "result diff"


if __name__ == "__main__":
    # test the kernel
    device = torch.device("cuda")
    sz = [256, 512, 1024, 2048, 4096, 8192]
    torch.manual_seed(42)
    for n in sz:
        for m in sz:
            print("#" * 100)
            print(f"n: {n}, m: {m}")
            a = torch.arange(n * m).reshape(n, m).contiguous().float().cuda()
            b = torch.randn(m, n).float().cuda()

            benchmark(transpose, a, b)
            b_my = torch.zeros_like(b)
            benchmark(lib.transpose_coalesced_read, a, b_my, prefix="transpose_coalesced_read")
            # print(b, b_my)
            diff_check(b, b_my, prefix="transpose_coalesced_read")
            benchmark(lib.transpose_coalesced_write, a, b_my, prefix="transpose_coalesced_write")
            diff_check(b, b_my, prefix="transpose_coalesced_write")

            benchmark(lib.transpose_smem, a, b_my, prefix="transpose_smem")
            diff_check(b, b_my, prefix="transpose_smem")
            benchmark(lib.transpose_smem_bcf, a, b_my, prefix="transpose_smem_bcf")
            diff_check(b, b_my, prefix="transpose_smem_bcf")
            benchmark(lib.transpose_smem_packed, a, b_my, prefix="transpose_smem_packed")
            diff_check(b, b_my, prefix="transpose_smem_packed")
            benchmark(lib.transpose_smem_swizzled_packed, a, b_my, prefix="transpose_smem_swizzled_packed")
            diff_check(b, b_my, prefix="transpose_smem_swizzled_bcf_packed")
