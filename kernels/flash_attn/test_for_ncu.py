import math
import time
from functools import partial
from typing import Optional

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

common_flags = ["-O3", "-std=c++17"]
# Load the CUDA kernel as a python module
lib = load(
    name="flash_attn",
    sources=["flash_attn.cu"],
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


def benchmark(op, q, k, v, o=None, warmup=0, rep=1, prefix="torch"):
    if o is not None:
        scale = 1 / math.sqrt(q.shape[-1])
        # warm up
        for i in range(warmup):
            op(q, k, v, o, scale)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(rep):
            op(q, k, v, o, scale)
        torch.cuda.synchronize()
    else:
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        # warm up
        for i in range(warmup):
            o = op(q, k, v)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(rep):
            o = op(q, k, v)
        torch.cuda.synchronize()
        o = o.transpose(1, 2).contiguous()

    duration = time.perf_counter() - start

    if prefix == "torch":
        # tflops ~= 2*b*q_h*s_q*s_kv*dim
        tflops = (
            2
            * q.shape[0]
            * q.shape[1]
            * q.shape[2]
            * k.shape[2]
            * v.shape[3]
            * rep
            / 1e12
            / duration
        )
        global baseline
        baseline = duration
        print(
            f"{prefix:40s} mean time: {duration / rep * 1000:8.6f} ms, {tflops:.2f} tflops"
        )
    else:
        # tflops ~= 2*b*q_h*s_q*s_kv*dim
        tflops = (
            2
            * q.shape[0]
            * q.shape[2]
            * q.shape[1]
            * k.shape[1]
            * v.shape[3]
            * rep
            / 1e12
            / duration
        )
        speedup = baseline / duration
        print(
            f"{prefix:40s} mean time: {duration / rep * 1000:8.6f} ms, speedup: {speedup:.2f}, tflops: {tflops:.2f}"
        )
    return o


def diff_check(a, b, prefix="torch", eps=0.016):
    if not torch.allclose(a, b, atol=eps, rtol=eps):
        diff = torch.abs(a - b)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        print(f"{prefix} result mean diff: {mean_diff:.6f}, max diff: {max_diff:.6f}")
    assert torch.allclose(a, b, atol=eps, rtol=eps), "result diff"


def test_all():
    print("#" * 100)
    print(f"prefill ")
    b = 1
    seq = 4096
    dim = 128
    head = 32
    q = torch.randn(b, seq, head, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, seq, head, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, seq, head, dim, device="cuda", dtype=torch.bfloat16)
    o = benchmark(
        partial(F.scaled_dot_product_attention, is_causal=True), q, k, v, prefix="torch"
    )

    o_my = torch.zeros_like(o)
    o_my = benchmark(lib.fmha_tma_128, q, k, v, o_my, prefix="fmha_tma_128")


if __name__ == "__main__":
    test_all()
