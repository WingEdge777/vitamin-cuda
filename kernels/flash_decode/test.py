import math
import time
from functools import partial
from typing import Optional

import flashinfer
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

common_flags = ["-O3", "-std=c++17"]
# Load the CUDA kernel as a python module
lib = load(
    name="flash_decode",
    sources=["flash_decode.cu"],
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

@torch.compile
def torch_native_decode(q, k, v, scale=None):
    # q: [head, dim] -> [32, 128]
    # k: [seq, head, dim] -> [4096, 32, 128]
    # v: [seq, head, dim] -> [4096, 32, 128]
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    # 调整维度以适应 Batched GEMV
    q_b = q.unsqueeze(1)  # [32, 1, 128]
    k_b = k.permute(1, 2, 0)  # [32, 128, 4096]
    v_b = v.transpose(0, 1)  # [32, 4096, 128]

    # S = Q @ K^T
    attn_scores = torch.matmul(q_b, k_b) * scale  # [32, 1, 4096]
    attn_probs = torch.softmax(attn_scores, dim=-1)

    # O = P @ V
    out = torch.matmul(attn_probs, v_b)  # [32, 1, 128]

    return out.squeeze(1)  # [32, 128]


def benchmark(op, q, k, v, o=None, warmup=10, rep=100, prefix="torch"):
    scale = 1 / math.sqrt(q.shape[-1])
    if o is not None:
        # warm up
        for i in range(warmup):
            op(q, k, v, o, scale)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(rep):
            op(q, k, v, o, scale)
        torch.cuda.synchronize()
    else:
        # warm up
        for i in range(warmup):
            o = op(q, k, v, scale)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(rep):
            o = op(q, k, v, scale)
        torch.cuda.synchronize()

    duration = time.perf_counter() - start
    # tflops ~= 2*q_h*s_q*s_kv*dim
    tflops = 2 * q.shape[0] * k.shape[0] * v.shape[2] * rep / 1e12 / duration

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
    return o


def diff_check(a, b, prefix="torch", eps=0.016):
    if not torch.allclose(a, b, atol=eps, rtol=eps):
        diff = torch.abs(a - b)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        print(f"{prefix} result mean diff: {mean_diff:.6f}, max diff: {max_diff:.6f}")
    assert torch.allclose(a, b, atol=eps, rtol=eps), "result diff"


def test_all():
    for seq in [4096, 8192, 8192 * 2]:
        dim = 128
        head = 32
        kv_head = 32
        print("#" * 100)
        print(f"prefill, kv seq: {seq}, head: {head}, dim: {dim}")
        q = torch.randn(head, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(seq, kv_head, dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(seq, kv_head, dim, device="cuda", dtype=torch.bfloat16)
        # flashinfer throw exception!
        # o = benchmark(
        #     flashinfer.single_decode_with_kv_cache, q, k, v, prefix="flash-infer"
        # )
        o = benchmark(torch_native_decode, q, k, v, prefix="torch")

        o_my = torch.zeros_like(o)
        o_my = benchmark(lib.flash_decode_tma_128, q, k, v, o_my, prefix="flash_decode_tma_128")
        diff_check(o, o_my, prefix="flash_decode_tma_128")


if __name__ == "__main__":
    test_all()
