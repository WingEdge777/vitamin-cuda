import time
from functools import partial
from pathlib import Path

import flashinfer
import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

common_flags = ["-O3", "-std=c++17"]
current_dir = Path(__file__).parent.resolve()
# Load the CUDA kernel as a python module
lib = load(
    name="sampling_lib",
    sources=[str(current_dir / "sampling.cu")],
    extra_cuda_cflags=common_flags
    + [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-Xptxas -v"
    ],
    extra_cflags=common_flags,
    verbose=True,
)

# vllm/vllm/v1/sample/ops/topk_topp_sampler.py
def apply_top_k_top_p_pytorch(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    allow_cpu_sync: bool = False,
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits.

    If a top-p is used, this function will sort the logits tensor,
    which can be slow for large batches.

    The logits tensor may be updated in-place.
    """
    if p is None:
        if k is None:
            return logits

        if allow_cpu_sync:
            # Avoid sorting vocab for top-k only case.
            return apply_top_k_only(logits, k)

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)

def random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Randomly sample from the probabilities.

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-GPU synchronization.
    """
    q = torch.empty_like(probs)
    # NOTE(woosuk): To batch-process the requests without their own seeds,
    # which is the common case, we first assume that every request does
    # not have its own seed. Then, we overwrite the values for the requests
    # that have their own seeds.
    if len(generators) != probs.shape[0]:
        q.exponential_()
    if generators:
        # TODO(woosuk): This can be slow because we handle each request
        # one by one. Optimize this.
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
    return probs.div_(q).argmax(dim=-1).view(-1)


def torch_topk_topp_sampling(logits, top_k, top_p, seed=42):
    k = torch.full((logits.shape[0],), top_k, dtype=torch.long, device=logits.device)
    p = torch.full((logits.shape[0],), top_p, dtype=torch.float32, device=logits.device)
    masked_logits = apply_top_k_top_p_pytorch(logits.clone(), k, p)
    probs = torch.softmax(masked_logits, dim=-1)
    generators = {0: torch.Generator(device="cuda")}
    generators[0].manual_seed(seed)
    return random_sample(probs, generators)


baseline = None


def benchmark(op, *args, warmup=10, rep=100, prefix="torch"):
    for _ in range(warmup):
        res = op(*args)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(rep):
        res = op(*args)
    torch.cuda.synchronize()
    duration = time.perf_counter() - start
    io_bytes = args[0].numel() * args[0].element_size() * rep
    bandwidth = io_bytes / duration / 1e9
    if prefix == "torch":
        global baseline
        baseline = duration
        print(f"{prefix:40s} mean time: {duration / rep * 1000:8.6f} ms, {bandwidth:.2f} GB/s")
    else:
        speedup = baseline / duration if baseline else 0.0
        print(f"{prefix:40s} mean time: {duration / rep * 1000:8.6f} ms, speedup: {speedup:.2f}, {bandwidth:.2f} GB/s")
    return res

def generate_realistic_logits(bs, vocab_size, num_spikes=5, device="cuda", dtype=torch.bfloat16):
    logits = torch.randn(bs, vocab_size, dtype=dtype, device=device)
    
    spike_indices = torch.randint(0, vocab_size, (bs, num_spikes), device=device)
    
    spike_values = 10.0 + 20.0 * torch.rand((bs, num_spikes), device=device, dtype=torch.float32)
    spike_values = spike_values.to(dtype)
    
    logits.scatter_(1, spike_indices, spike_values)
    
    return logits

def test():
    top_k = 20
    top_p = 0.95
    seed = 42
    step = 1
    ns = [1, 8, 16, 32]
    ms = [128000, 256000, 320000]
    for bs in ns:
        for vocab_size in ms:
            print("#" * 100)
            print(f"bs: {bs}, vocab_size: {vocab_size}")
            logits = generate_realistic_logits(bs, vocab_size, num_spikes=50)
            res = benchmark(torch_topk_topp_sampling, logits, top_k, top_p, seed, prefix="torch")
            res = benchmark(partial(flashinfer.sampling.top_k_top_p_sampling_from_logits, seed=seed, offset=step), logits, top_k, top_p, prefix="flashinfer")
            res = benchmark(lib.sampling_topk_topp_batched, logits, top_k, top_p, seed, step, prefix="sampling_topk_topp_batched")


if __name__ == "__main__":
    test()
