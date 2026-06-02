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
    name="relu_lib",
    sources=[str(current_dir / "relu.cu")],
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
    global baseline
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
        baseline = avg_duration
        print(f"{prefix:30s} mean time: {avg_duration:8.6f} ms")
    else:
        speedup = baseline / avg_duration
        print(f"{prefix:30s} mean time: {avg_duration:8.6f} ms, speedup: {speedup:.2f}")


def diff_check(a, b, prefix="torch", eps=1e-5):
    message = f"{prefix} result diff"
    assert torch.max(torch.abs(a - b)).item() < eps, message


if __name__ == "__main__":
    # test the kernel
    device = torch.device("cuda")
    sz = [1024, 2048, 4096]
    for n in sz:
        for m in sz:
            print("#" * 100)
            print(f"n: {n}, m: {m}")
            a = torch.randn(n, m).float().cuda()
            b = torch.empty_like(a).float().cuda()

            benchmark(partial(torch.clamp, min=0, out=b), a)
            b_my = torch.empty_like(a)

            benchmark(lib.relu, a, b_my, prefix="relu")
            # print(b, b_my)
            diff_check(b, b_my, prefix="relu")
            benchmark(lib.relu_fp32x4, a, b_my, prefix="relu_fp32x4")
            diff_check(b, b_my, prefix="relu_fp32x4")

            ################### half
            a = a.half()
            b = b.half()
            benchmark(partial(torch.clamp, min=0, out=b), a)
            b_my = b_my.half()

            benchmark(lib.relu, a, b_my, prefix="relu_half")
            diff_check(b, b_my, prefix="relu_half", eps=1e-3)
            benchmark(lib.relu_fp16x2, a, b_my, prefix="relu_fp16x2")
            diff_check(b, b_my, prefix="relu_fp16x2", eps=1e-3)
            benchmark(lib.relu_fp16x8, a, b_my, prefix="relu_fp16x8")
            diff_check(b, b_my, prefix="relu_fp16x8", eps=1e-3)
            benchmark(
                lib.relu_fp16x8_packed,
                a,
                b_my,
                prefix="relu_fp16x8_packed",
            )
            diff_check(b, b_my, prefix="relu_fp16x8_packed", eps=1e-3)
