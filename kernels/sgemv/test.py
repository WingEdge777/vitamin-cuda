import numpy as np
from functools import partial
from pathlib import Path

import torch
from flashinfer.testing.utils import bench_gpu_time
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

common_flags = ["-O3", "-std=c++17"]
current_dir = Path(__file__).parent.resolve()
# Load the CUDA kernel as a python module
lib = load(
    name="gemv_lib",
    sources=[str(current_dir / "sgemv.cu")],
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


def diff_check(a, b, prefix="torch", eps=1e-4):
    if not torch.allclose(a, b, atol=eps, rtol=eps):
        print(f"{prefix} result diff: {torch.mean(torch.abs(a - b)).item()}")
    assert torch.allclose(a, b, atol=eps, rtol=eps), "result diff"


if __name__ == "__main__":
    # test the kernel
    nm = [512, 1024, 2048, 4096, 8192]
    torch.manual_seed(42)
    for n in nm:
        for m in nm:
            print("#" * 100)
            print(f"n: {n}, m: {m}")
            # a dot b = c
            a = torch.randn(n, m).float().cuda()
            b = torch.randn(m, 1).float().cuda()
            c = torch.zeros(n, 1).cuda()

            benchmark(partial(torch.matmul, out=c), a, b)
            c_my = torch.zeros_like(c)

            benchmark(lib.gemv, a, b, c_my, prefix="gemv")
            diff_check(c, c_my, prefix="gemv")
            benchmark(lib.gemv_fp32x4, a, b, c_my, prefix="gemv_fp32x4")
            # print(c.flatten(), c_my.flatten())
            diff_check(c, c_my, prefix="gemv_fp32x4")
