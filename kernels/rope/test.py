import time
from functools import partial
from typing import Optional

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

common_flags = ["-O3", "-std=c++17"]
# Load the CUDA kernel as a python module
lib = load(
    name="rope_lib",
    sources=["rope.cu"],
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
base = 10000


def compute_default_rope_parameters(head_dim):
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2).float().cuda() / head_dim)
    )  # 64
    return inv_freq


INV_FREQS = {
    256: compute_default_rope_parameters(256),
    128: compute_default_rope_parameters(128),
}
position_ids = torch.arange(8192).float().cuda()
freqs = {
    256:torch.cat([torch.outer(position_ids, INV_FREQS[256]), torch.outer(position_ids, INV_FREQS[256])], dim=-1),
    128:torch.cat([torch.outer(position_ids, INV_FREQS[128]), torch.outer(position_ids, INV_FREQS[128])], dim=-1),
}
COS = {
    256:torch.cos(freqs[256]),
    128:torch.cos(freqs[128]),
}
SIN = {
    256:torch.sin(freqs[256]),
    128:torch.sin(freqs[128]),
}

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed

def rope_with_sin_cos_cache(q):  # q shape: [seqlen, head_dim]
    # inv_freq = compute_default_rope_parameters(q.shape[1])
    # position_ids = torch.arange(q.shape[0], device=q.device).float()

    # # [seq_len] outer [dim/2] -> [seq_len, dim/2]
    # freqs = torch.outer(position_ids, inv_freq)

    # # [seq_len, dim/2] -> [seq_len, dim]
    # freqs = torch.cat([freqs, freqs], dim=-1)

    # cos, sin = torch.cos(freqs), torch.sin(freqs)
    cos = COS[q.shape[1]][:q.shape[0], :q.shape[1]]
    sin = SIN[q.shape[1]][:q.shape[0], :q.shape[1]]

    return apply_rotary_pos_emb(q, cos, sin)

# neo-x stype rope, single head single batch
def rope(q):  # q shape: [seqlen, head_dim]
    inv_freq = compute_default_rope_parameters(q.shape[1])
    position_ids = torch.arange(q.shape[0], device=q.device).float()

    # [seq_len] outer [dim/2] -> [seq_len, dim/2]
    freqs = torch.outer(position_ids, inv_freq)

    # [seq_len, dim/2] -> [seq_len, dim]
    freqs = torch.cat([freqs, freqs], dim=-1)

    cos, sin = torch.cos(freqs), torch.sin(freqs)
    cos = COS[q.shape[1]][:q.shape[0], :q.shape[1]]
    sin = SIN[q.shape[1]][:q.shape[0], :q.shape[1]]

    return apply_rotary_pos_emb(q, cos, sin)


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

            b = benchmark(rope, a)
            c = benchmark(rope_with_sin_cos_cache, a, prefix="torch.rope_with_sin_cos_cache")
            diff_check(b, c, prefix="rope_with_sin_cos_cache")
            b_my = torch.empty_like(a)
            benchmark(lib.rope, a, b_my, prefix="rope")
            # print(b, b_my)
            diff_check(b, b_my, prefix="rope")

            benchmark(lib.rope_fp32x4, a, b_my, prefix="rope_fp32x4")
            diff_check(b, b_my, prefix="rope_fp32x4")
