import math
import numpy as np
from functools import partial
from pathlib import Path

import flashinfer
import torch
from flashinfer.testing.utils import bench_gpu_time
from torch.nn import functional as F
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

common_flags = ["-O3", "-std=c++17"]
current_dir = Path(__file__).parent.resolve()
# Load the CUDA kernel as a python module
lib = load(
    name="flash_decode",
    sources=[str(current_dir / "flash_decode.cu")],
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
    extra_ldflags=["-L/usr/local/cuda/lib64/stubs", "-lcuda"],
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

    if prefix == "flash-infer":
        input_args = (q, k, v)
    elif o is not None:
        input_args = (q, k, v, o, scale)
    else:
        input_args = (q, k, v, scale)

    times = bench_gpu_time(
        fn=op,
        input_args=input_args,
        dry_run_iters=warmup,
        repeat_iters=rep,
        enable_cupti=False,
        use_cuda_graph=False,
        cold_l2_cache=True,
    )
    avg_duration = float(np.median(times))

    if prefix == "flash-infer":
        o = op(q, k, v)
    elif o is None:
        o = op(q, k, v, scale)

    io_bytes = (q.numel() + k.numel() + v.numel() + o.numel()) * q.element_size()
    bandwidth = io_bytes / (avg_duration / 1000) / 1e9

    if prefix == "torch":
        global baseline
        baseline = avg_duration
        prefix = "torch.compile"
        print(f"{prefix:40s} mean time: {avg_duration:8.6f} ms, {bandwidth:.2f} GB/s")
    else:
        speedup = baseline / avg_duration
        print(f"{prefix:40s} mean time: {avg_duration:8.6f} ms, speedup: {speedup:.2f}, GB/s: {bandwidth:.2f}")
    return o


def diff_check(a, b, prefix="torch", eps=0.016):
    if not torch.allclose(a, b, atol=eps, rtol=eps):
        diff = torch.abs(a - b)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        print(f"{prefix} result mean diff: {mean_diff:.6f}, max diff: {max_diff:.6f}")
    assert torch.allclose(a, b, atol=eps, rtol=eps), "result diff"


def test_all():
    for seq in [8192, 1024 * 16, 1024 * 32, 1024 * 64, 1024 * 128, 1024 * 128 + 1]:
        dim = 128
        head = 32
        kv_head = 32
        print("#" * 100)
        print(f"decode, kv seq: {seq}, head: {head}, dim: {dim}")
        q = torch.randn(head, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(seq, kv_head, dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(seq, kv_head, dim, device="cuda", dtype=torch.bfloat16)

        o = benchmark(torch_native_decode, q, k, v, prefix="torch")
        # flashinfer throw exception!
        o = benchmark(
            flashinfer.single_decode_with_kv_cache, q, k, v, prefix="flash-infer"
        )

        o_my = torch.zeros_like(o)
        o_my = benchmark(
            lib.flash_decode_tma_128, q, k, v, o_my, prefix="flash_decode_tma_128"
        )
        diff_check(o, o_my, prefix="flash_decode_tma_128")
        o_my = benchmark(
            lib.flash_decode_tma_dbf_k_128, q, k, v, o_my, prefix="flash_decode_tma_dbf_k_128"
        )
        diff_check(o, o_my, prefix="flash_decode_tma_dbf_k_128")


if __name__ == "__main__":
    test_all()
