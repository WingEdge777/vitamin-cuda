from functools import partial
from pathlib import Path

import numpy as np
import torch
from flashinfer.testing.utils import bench_gpu_time, get_l2_cache_size
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

common_flags = ["-O3", "-std=c++17"]
current_dir = Path(__file__).parent.resolve()
# Load the CUDA kernel as a python module
lib = load(
    name="cmsum_lib",
    sources=[str(current_dir / "cmsum.cu")],
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


baseline = None


def benchmark(op, a, b, warmup=10, rep=100, prefix="torch"):
    input_args = (a, b)

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

    return op(a, b)


def diff_check(a, b, prefix="torch", eps=1e-4):
    message = f"{prefix} result diff"
    assert torch.mean(torch.abs(a - b)).item() < eps, message


if __name__ == "__main__":
    # test the kernel
    device = torch.device("cuda")
    sz = [2048, 4096, 8192]
    torch.manual_seed(42)
    for n in sz:
        print("#" * 100)
        print(f"n: {n}")
        a = torch.randn(n).float().cuda()
        b = torch.randn(n).float().cuda()

        c = benchmark(partial(torch.cumsum(out=b)), a)
        c_my = benchmark(lib.cmsum, a, b, prefix="cmsum")

        diff_check(c, c_my, prefix="cmsum")

        c_my = benchmark(lib.cmsum_fp32x4, a, b, prefix="cmsum_fp32x4")
        diff_check(c, c_my, prefix="cmsum_fp32x4")
