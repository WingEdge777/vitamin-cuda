import time
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

common_flags = ["-O3", "-std=c++17"]
# Load the CUDA kernel as a python module
lib = load(
    name="embedding_lib",
    sources=["embedding.cu"],
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


def benchmark(op, a, b, c=None, warmup=10, rep=500, prefix="torch"):
    if c is not None:
        # warm up
        for i in range(warmup):
            op(a, b, c)
        torch.cuda.synchronize()
        start = time.time()
        for i in range(rep):
            op(a, b, c)
        torch.cuda.synchronize()
        print(
            f"{prefix:30s} mean time: {(time.time() - start) / rep * 1000:.6f} ms"
        )
    else:
        # warm up
        for i in range(warmup):
            op(a, b)
        torch.cuda.synchronize()
        start = time.time()
        for i in range(rep):
            op(a, b)
        torch.cuda.synchronize()
        print(
            f"{prefix:30s} mean time: {(time.time() - start) / rep * 1000:.6f} ms"
        )


def diff_check(a, b, prefix="torch", eps=1e-5):
    diff_val = torch.max(torch.abs(a - b)).item()

    # 2. 在 message 里把这个变量打印出来
    #    :.6f 表示保留6位小数，让你看清楚
    message = f"{prefix} result diff! Actual: {diff_val:.6f} >= Eps: {eps}"

    # 3. 断言
    assert diff_val < eps, message


if __name__ == "__main__":
    # test the kernel
    device = torch.device("cuda")
    # 涵盖了 搜广推特征工程 到 llm 级别的 embedding 大小和特征长度
    emd_sizes = [32, 128, 1024, 16384, 10240, 102400]
    emd_dims = [8, 32, 128, 512, 1024, 2048, 4096]
    seq_lens = [1, 64, 256, 1024, 4096]
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
