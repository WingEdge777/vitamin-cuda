import numpy as np
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from flashinfer.testing.utils import bench_gpu_time
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

common_flags = ["-O3", "-std=c++17"]
current_dir = Path(__file__).parent.resolve()
# Load the CUDA kernel as a python module
lib = load(
    name="embedding_lib",
    sources=[str(current_dir / "embedding.cu")],
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


def benchmark(op, a, b, c=None, warmup=10, rep=100, prefix="torch"):
    input_args = (a, b, c) if c is not None else (a, b)

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

    if prefix == "torch":
        global baseline
        baseline = avg_duration
        print(f"{prefix:30s} mean time: {avg_duration:8.6f} ms")
    else:
        speedup = baseline / avg_duration
        print(f"{prefix:30s} mean time: {avg_duration:8.6f} ms, speedup: {speedup:.2f}")


def diff_check(a, b, prefix="torch", eps=1e-5):
    diff_val = torch.max(torch.abs(a - b)).item()

    message = f"{prefix} result diff! Actual: {diff_val:.6f} >= Eps: {eps}"

    assert diff_val < eps, message


if __name__ == "__main__":
    # test the kernel
    device = torch.device("cuda")
    # 涵盖了 搜广推特征工程 到 llm 级别的 embedding 大小和特征长度
    emd_sizes = [32, 128, 1024, 102400]
    emd_dims = [32, 128, 512]
    seq_lens = [1, 64, 256, 1024]
    import itertools
    for emd_size, emd_dim in itertools.product(emd_sizes, emd_dims):
        for seq_len in seq_lens:

            print("#" * 100)
            print(f"emd_size: {emd_size}, emd_dim: {emd_dim}, seq_len: {seq_len}")
            emd_idx = torch.randint(0, emd_size, (seq_len,), dtype=torch.int32, device=device)
            weight = torch.randn(emd_size, emd_dim, dtype=torch.float32, device=device)

            benchmark(F.embedding, emd_idx, weight)
            out_my = torch.empty(seq_len, emd_dim, dtype=torch.float32, device=device)

            benchmark(lib.embedding, emd_idx, weight, out_my, prefix="embedding")
            out = F.embedding(emd_idx, weight=weight)
            diff_check(out, out_my, prefix="embedding")
            benchmark(lib.embedding_fp32x4, emd_idx, weight, out_my, prefix="embedding_fp32x4")
            out = F.embedding(emd_idx, weight=weight)
            diff_check(out, out_my, prefix="embedding_fp32x4")
            benchmark(lib.embedding_fp32x4_packed, emd_idx, weight, out_my, prefix="embedding_fp32x4_packed")
            out = F.embedding(emd_idx, weight=weight)
            diff_check(out, out_my, prefix="embedding_fp32x4_packed")

            ################### half
            emd_idx = emd_idx
            weight = weight.half()
            benchmark(F.embedding, emd_idx, weight)
            out_my = out_my.half()

            benchmark(lib.embedding, emd_idx, weight, out_my, prefix="embedding_half")
            out = F.embedding(emd_idx, weight=weight)
            diff_check(out, out_my, prefix="embedding_half", eps=1e-3)
            benchmark(lib.embedding_fp16x8, emd_idx, weight, out_my, prefix="embedding_fp16x8")
            out = F.embedding(emd_idx, weight=weight)
            diff_check(out, out_my, prefix="embedding_fp16x8", eps=1e-3)
            benchmark(
                lib.embedding_fp16x8_packed,
                emd_idx,
                weight,
                out_my,
                prefix="embedding_fp16x8_packed",
            )
            out = F.embedding(emd_idx, weight=weight)
            diff_check(out, out_my, prefix="embedding_fp16x8_packed", eps=1e-3)
