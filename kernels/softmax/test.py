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
    name="softmax_lib",
    sources=[str(current_dir / "softmax.cu")],
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


def diff_check(a, b, prefix="torch", eps=1e-3):
    if not torch.allclose(a, b, atol=eps, rtol=eps):
        print(f"{prefix} result diff: {torch.mean(torch.abs(a - b)).item()}")
    assert torch.allclose(a, b, atol=eps, rtol=eps), "result diff"


def test_small():
    bs = [128, 256, 1024, 2048]
    sz = [1024, 2048, 4096, 8192]
    torch.manual_seed(42)
    for n in bs:
        for m in sz:
            print("#" * 100)
            print(f"bs: {n}, hidden_size: {m}")
            a = torch.randn(n, m).float().cuda()
            b = torch.empty(n, m).float().cuda()

            # benchmark(partial(torch.softmax, dim=1, out=b), a)
            b_my = torch.empty_like(a)
            # benchmark(lib.softmax, a, b_my, prefix="softmax")
            # # print(b, b_my)
            # diff_check(b, b_my, prefix="softmax")

            # benchmark(lib.softmax_fp32x4, a, b_my, prefix="softmax_fp32x4")
            # # print(b, b_my)
            # diff_check(b, b_my, prefix="softmax_fp32x4")

            a = a.half()
            b = b.half()
            b_my = b_my.half()
            benchmark(partial(torch.softmax, dim=1, out=b), a)
            benchmark(lib.softmax, a, b_my, prefix="softmax_fp16")
            diff_check(b, b_my, prefix="softmax_fp16")

            benchmark(
                lib.softmax_fp16x8_packed, a, b_my, prefix="softmax_fp16x8_packed"
            )
            diff_check(b, b_my, prefix="softmax_fp16x8_packed")

            benchmark(lib.softmax_medium, a, b_my, prefix="softmax_medium")
            diff_check(b, b_my, prefix="softmax_medium")

            benchmark(lib.softmax_extreme, a, b_my, prefix="softmax_extreme")
            diff_check(b, b_my, prefix="softmax_extreme")

            benchmark(lib.softmax_arbitrary, a, b_my, prefix="softmax_arbitrary")
            diff_check(b, b_my, prefix="softmax_arbitrary")


def test_large():
    ns = [4]
    ms = [
        8192 * 2,
        32768,
        8192 * 8,
        114688,
        8192 * 32,
        8192 * 128,
        8192 * 1024,
        8192 * 4096,
    ]
    for n in ns:
        for m in ms:
            print("#" * 100)
            print(f"bs: {n}, hidden_size: {m}")
            a = torch.randn(n, m).half().cuda()
            b = torch.zeros(n, m).half().cuda()

            benchmark(partial(torch.softmax, dim=1, out=b), a)
            b_my = torch.zeros_like(a)

            if m <= 32768:
                benchmark(lib.softmax_medium, a, b_my, prefix="softmax_medium")
                diff_check(b, b_my, prefix="softmax_medium")
            if m <= 114688:
                benchmark(lib.softmax_extreme, a, b_my, prefix="softmax_extreme")
                diff_check(b, b_my, prefix="softmax_extreme")

            benchmark(lib.softmax_arbitrary, a, b_my, prefix="softmax_arbitrary")
            diff_check(b, b_my, prefix="softmax_arbitrary")

            benchmark(lib.softmax_splitk, a, b_my, prefix="softmax_splitk")
            diff_check(b, b_my, prefix="softmax_splitk")


def test_run():
    ns = [4]
    ms = [8192 * 2]
    for n in ns:
        for m in ms:
            print("#" * 100)
            print(f"bs: {n}, hidden_size: {m}")
            a = torch.randn(n, m).half().cuda()
            b = torch.zeros(n, m).half().cuda()

            benchmark(partial(torch.softmax, dim=1, out=b), a)
            b_my = torch.zeros_like(a)

            if m <= 32768:
                benchmark(lib.softmax_medium, a, b_my, prefix="softmax_medium")
                diff_check(b, b_my, prefix="softmax_medium")
            if m <= 114688:
                benchmark(lib.softmax_extreme, a, b_my, prefix="softmax_extreme")
                diff_check(b, b_my, prefix="softmax_extreme")

            benchmark(lib.softmax_arbitrary, a, b_my, prefix="softmax_arbitrary")
            diff_check(b, b_my, prefix="softmax_arbitrary")

            benchmark(lib.softmax_splitk, a, b_my, prefix="softmax_splitk")
            diff_check(b, b_my, prefix="softmax_splitk")


if __name__ == "__main__":
    # test the kernel
    test_small()
    # test_large()
    # test_run()
