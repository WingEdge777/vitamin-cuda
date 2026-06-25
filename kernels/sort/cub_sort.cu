#include <cstdint>

#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

void cub_sort_bf16(__nv_bfloat16 *d_in, __nv_bfloat16 *d_out, int n, cudaStream_t stream) {
    const auto device = at::cuda::current_device();
    size_t temp_storage_bytes = 0;

    cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_bytes, d_in, d_out, n, 0, 16, stream);

    auto temp_storage = torch::empty({static_cast<int64_t>(temp_storage_bytes)},
                                     torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device));

    cub::DeviceRadixSort::SortKeys(temp_storage.data_ptr<uint8_t>(), temp_storage_bytes, d_in, d_out, n, 0, 16, stream);
}

void cub_sort(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "All tensors must be on CUDA");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "Tensors must be contiguous");
    TORCH_CHECK(a.scalar_type() == torch::kBFloat16, "cub_sort only supports bfloat16");
    TORCH_CHECK(b.scalar_type() == torch::kBFloat16, "cub_sort only supports bfloat16");
    TORCH_CHECK(a.sizes() == b.sizes(), "Input and output must have the same shape");

    const int n = static_cast<int>(a.numel());
    if (n == 0) {
        return;
    }

    cub_sort_bf16(reinterpret_cast<__nv_bfloat16 *>(a.data_ptr()),
                  reinterpret_cast<__nv_bfloat16 *>(b.data_ptr()),
                  n,
                  at::cuda::getCurrentCUDAStream());
}
