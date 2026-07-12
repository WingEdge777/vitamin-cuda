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
    name="cumsum_lib",
    sources=[str(current_dir / "cumsum.cu")],
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


def benchmark(op, a, b=None, warmup=10, rep=100, prefix="torch"):

    input_args = (a, b) if b is not None else (a,)

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

    return op(*input_args)


def diff_check(a, b, prefix="torch", eps=1e-3):
    if not torch.allclose(a, b, atol=eps, rtol=eps):
        diff = torch.abs(a - b)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        print(f"{prefix} result mean diff: {mean_diff:.6f}, max diff: {max_diff:.6f}")
    assert torch.allclose(a, b, atol=eps, rtol=eps), "result diff"


if __name__ == "__main__":
    # test the kernel
    device = torch.device("cuda")
    bs = [1, 32, 64, 128]
    sz = [2048, 4096, 8192, 12800]
    torch.manual_seed(42)
    for n in bs:
        for m in sz:
            print("#" * 100)
            print(f"n: {n}, m: {m}")
            a = torch.randn(n, m).float().cuda()
            b = torch.randn(n, m).float().cuda()

            benchmark(partial(torch.cumsum, dim=-1, out=b), a)

            b_my = torch.zeros_like(b)
            benchmark(lib.cumsum_fp32, a, b_my, prefix="cumsum_fp32")
            diff_check(b, b_my, prefix="cumsum_fp32")

            b_my = torch.zeros_like(b)
            benchmark(lib.cumsum_fp32x4, a, b_my, prefix="cumsum_fp32x4")
            diff_check(b, b_my, prefix="cumsum_fp32x4")
