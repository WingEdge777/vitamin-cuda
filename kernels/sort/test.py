import time
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from flashinfer.testing.utils import bench_gpu_time, get_l2_cache_size
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

common_flags = ["-O3", "-std=c++17"]
current_dir = Path(__file__).parent.resolve()
# Load the CUDA kernel as a python module
lib = load(
    name="sort_lib",
    sources=[str(current_dir / "sort.cu"), str(current_dir / "cub_sort.cu")],
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


def diff_check(a, b, prefix="torch"):
    assert torch.max(torch.abs(a - b)).item() < 1e-5, f"{prefix} result diff"


if __name__ == "__main__":
    # test the kernel
    device = torch.device("cuda")
    sz = [1024, 2048, 4096]
    for n in sz:
        print("#" * 100)
        print(f"n: {n}")
        a = torch.randn(n).bfloat16().cuda()
        b = torch.empty(n).bfloat16().cuda()

        benchmark(lib.cub_sort, a, b)
        b_my = torch.zeros_like(b)
        benchmark(lib.sort, a, b, prefix="sort")
        diff_check(b, b_my, prefix="sort")
