from functools import partial
import time
import torch
import tilelang as tl
import tilelang.language as T
tl.set_log_level("warning")
import sys

sys.path.append("../../../")
from kernels.softmax.test import lib, benchmark


# make sure dim are power of 2
@tl.jit()
def softmax_tilelang(bs, dim, dtype: T.dtype = T.float32):
    block_size = 256
    accum_dtype = T.float32
    tiling_size = min(tl.next_power_of_2(dim), 8192)
    tiling_num = tl.cdiv(dim, tiling_size)

    @T.prim_func
    def main(
        X: T.Tensor([bs, dim], dtype),
        Y: T.Tensor([bs, dim], dtype),
    ):
        with T.Kernel(bs, threads=block_size) as bx:
            x = T.alloc_fragment([tiling_size], dtype)
            y = T.alloc_fragment([tiling_size], dtype)

            x_exp = T.alloc_fragment([tiling_size], accum_dtype)

            m_local = T.alloc_fragment([1], accum_dtype)
            d_local = T.alloc_fragment([1], accum_dtype)
            m_global = T.alloc_fragment([1], accum_dtype)
            d_global = T.alloc_fragment([1], accum_dtype)

            m_global[0] = T.min_value(accum_dtype)
            d_global[0] = 0.0

            for by in T.Pipelined(0, tiling_num):
                T.copy(X[bx, by * tiling_size : (by + 1) * tiling_size], x)

                T.reduce_max(x, m_local, dim=0, clear=True)
                for y_id in T.Parallel(tiling_size):
                    x_exp[y_id] = T.exp(x[y_id] - m_local[0])
                T.reduce_sum(x_exp, d_local, dim=0, clear=True)

                # update m_global, sglobal
                new_m = T.max(m_global[0], m_local[0])
                d_global[0] = d_global[0] * T.exp(m_global[0] - new_m) + d_local[0] * T.exp(m_local[0] - new_m)
                m_global[0] = new_m

            for by in T.Pipelined(0, tiling_num):
                T.copy(X[bx, by * tiling_size : (by + 1) * tiling_size], x)

                for i in T.Parallel(tiling_size):
                    y[i] = T.exp(x[i] - m_global[0]) / d_global[0]

                T.copy(y, Y[bx, by * tiling_size : (by + 1) * tiling_size])

    return main


def diff_check(a, b, prefix="torch", eps=1e-3):
    if not torch.allclose(a, b, atol=eps, rtol=eps):
        diff = torch.abs(a - b)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        print(f"{prefix} result mean diff: {mean_diff:.6f}, max diff: {max_diff:.6f}")
    assert torch.allclose(a, b, atol=eps, rtol=eps), "result diff"

def test_small():
    bs = [256, 1024]
    sz = [1024, 4096, 8192]
    torch.manual_seed(42)
    for n in bs:
        for m in sz:
            print("#" * 100, "test_small")
            print(f"bs: {n}, hidden_size: {m}")
            a = torch.randn(n, m).float().cuda()
            b = torch.empty(n, m).float().cuda()

            benchmark(partial(torch.softmax, dim=1, out=b), a)
            b_my = torch.empty_like(a)

            benchmark(lib.softmax_fp32x4, a, b_my, prefix="softmax_fp32x4")
            # print(b, b_my)
            diff_check(b, b_my, prefix="softmax_fp32x4")

            kernel = softmax_tilelang(a.shape[0], a.shape[1], dtype=T.float32)
            benchmark(kernel, a, b_my, prefix="tl.float32")
            diff_check(b, b_my, prefix="tl.float32")

            a = a.half()
            b = b.half()
            b_my = b_my.half()
            benchmark(partial(torch.softmax, dim=1, out=b), a)
            benchmark(lib.softmax_arbitrary, a, b_my, prefix="softmax_arbitrary")
            diff_check(b, b_my, prefix="softmax_arbitrary")
            kernel = softmax_tilelang(a.shape[0], a.shape[1], dtype=T.float16)
            benchmark(kernel, a, b_my, prefix="tl.float16")
            diff_check(b, b_my, prefix="tl.float16")


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
            print("#" * 100, "test_large")
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
            kernel = softmax_tilelang(a.shape[0], a.shape[1], dtype=T.float16)
            benchmark(kernel, a, b_my, prefix="tl.float16")
            diff_check(b, b_my, prefix="tl.float16")


if __name__ == "__main__":
    test_small()
    # test_large()
