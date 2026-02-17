// adapted from https://github.com/xlite-dev/LeetCUDA/blob/main/kernels/sgemm/sgemm_cublas.cu
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/types.h>

#include "cublas_v2.h"

void cublas_sgemm(float *A, float *B, float *C, size_t M, size_t N, size_t K, bool use_tf32) {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    static float alpha = 1.0;
    static float beta = 0.0;
    if (use_tf32) {
        cublasGemmEx(handle,
                     CUBLAS_OP_N,
                     CUBLAS_OP_N,
                     N,
                     M,
                     K,
                     &alpha,
                     B,
                     CUDA_R_32F,
                     N,
                     A,
                     CUDA_R_32F,
                     K,
                     &beta,
                     C,
                     CUDA_R_32F,
                     N,
                     CUBLAS_COMPUTE_32F_FAST_TF32,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    } else {
        cublasGemmEx(handle,
                     CUBLAS_OP_N,
                     CUBLAS_OP_N,
                     N,
                     M,
                     K,
                     &alpha,
                     B,
                     CUDA_R_32F,
                     N,
                     A,
                     CUDA_R_32F,
                     K,
                     &beta,
                     C,
                     CUDA_R_32F,
                     N,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT);
    }
}

void sgemm_cublas(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(), "All tensors must be on CUDA");
    TORCH_CHECK(a.is_contiguous(), "Tensor A must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "Tensor B must be contiguous");
    TORCH_CHECK(c.is_contiguous(), "Tensor C must be contiguous");

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    cublas_sgemm(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K, false);
}

void sgemm_cublas_tf32(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(), "All tensors must be on CUDA");
    TORCH_CHECK(a.is_contiguous(), "Tensor A must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "Tensor B must be contiguous");
    TORCH_CHECK(c.is_contiguous(), "Tensor C must be contiguous");

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    cublas_sgemm(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K, true);
}
