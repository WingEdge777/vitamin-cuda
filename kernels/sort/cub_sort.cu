#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/types.h>

void cub_sort(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "All tensors must be on CUDA");
    TORCH_CHECK(a.is_contiguous(), "Tensor A must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "Tensor B must be contiguous");

    const int N = a.size(0);
    // TODO
}
