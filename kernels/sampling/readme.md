# LLM Generattion Sampling Kernels

- [ ] sampling_topk_topp_batched
- [ ] sampling_topk_topp_split_k

## Test

```bash
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
python test.py
```

## Sample output

```yaml
```
