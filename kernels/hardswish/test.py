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
    name="hardswish_lib",
    sources=[str(current_dir / "hardswish.cu")],
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


def diff_check(a, b, prefix="torch", eps=1e-5):
    message = f"{prefix} result diff"
    # print("mean diff : ", torch.mean(torch.abs(a - b)).item())
    assert torch.mean(torch.abs(a - b)).item() < eps, message


if __name__ == "__main__":
    # test the kernel
    device = torch.device("cuda")
    sz = [1024, 2048, 4096]
    for n in sz:
        for m in sz:
            print("#" * 100)
            print(f"n: {n}, m: {m}")
            a = torch.randn(n, m).float().cuda()
            b = a.clone()

            benchmark(partial(F.hardswish, inplace=True), b)
            b_my = torch.empty_like(a)

            benchmark(lib.hardswish, a, b_my, prefix="hardswish")
            b = F.hardswish(a)
            diff_check(b, b_my, prefix="hardswish")
            benchmark(lib.hardswish_fp32x4, a, b_my, prefix="hardswish_fp32x4")
            b = F.hardswish(a)
            diff_check(b, b_my, prefix="hardswish_fp32x4")

            ################### half
            a = a.half()
            b = a.clone()
            benchmark(partial(F.hardswish, inplace=True), b)
            b_my = b_my.half()

            benchmark(lib.hardswish, a, b_my, prefix="hardswish_half")
            b = F.hardswish(a)
            diff_check(b, b_my, prefix="hardswish_half", eps=5e-3)
            benchmark(lib.hardswish_fp16x2, a, b_my, prefix="hardswish_fp16x2")
            b = F.hardswish(a)
            diff_check(b, b_my, prefix="hardswish_fp16x2", eps=5e-3)
            benchmark(lib.hardswish_fp16x8, a, b_my, prefix="hardswish_fp16x8")
            b = F.hardswish(a)
            diff_check(b, b_my, prefix="hardswish_fp16x8", eps=5e-3)
            benchmark(
                lib.hardswish_fp16x8_packed,
                a,
                b_my,
                prefix="hardswish_fp16x8_packed",
            )
            b = F.hardswish(a)
            diff_check(b, b_my, prefix="hardswish_fp16x8_packed", eps=1e-3)
