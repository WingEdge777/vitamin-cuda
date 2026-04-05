#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/types.h>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])

#define SWIZZLE_A(x, y) ((y) ^ ((x >> 2) << 3))

const int WARP_SIZE = 32;

// gemm fp32
__global__ void sgemm_naive_kernel(float *a, float *b, float *c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.f;
    if (row < m && col < n) {
        for (int i = 0; i < k; i++) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

// a block calculate c[128][128], each thread c[8][8]
template <const int BM = 128, const int BN = 128, const int BK = 16, const int TM = 8, const int TN = 8>
__global__ void sgemm_tiling_kernel(float *a, float *b, float *c, int m, int n, int k) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x; // 0~255; 8 warps, 2x4 warp tiling; each warp 8x4 thread tiling
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // Each block loads 64x16 of A and 8x128 of B per pass, two passes → 128x16 and 16x128
    // Every 4 threads load one row of A (16 elems); every 32 threads load one row of B (128 elems)
    int load_a_row = tid / 4;               // 0~63
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12...
    int load_b_row = tid / WARP_SIZE;       // 0~8
    int load_b_col = (tid % WARP_SIZE) * 4; // 0,4,8,12,16,20,24,28...

    // Warp tiling: every 4 warps cover the upper/lower 64x128 halves of C
    int warp_row = warp_id / 4;      // 0, 1
    int warp_col = warp_id % 4;      // 0, 1, 2, 3
    int t_row_in_warp = lane_id / 4; // 0~7
    int t_col_in_warp = lane_id % 4; // 0~3

    // C output origin: each thread computes an 8x8 tile, 256 threads × 64 = 128×128
    int c_row = warp_row * 64 + t_row_in_warp * 8;
    int c_col = warp_col * 32 + t_col_in_warp * 8;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float sum[TM][TN] = {0.f};

    for (int bk = 0; bk < k; bk += BK) {
        FLOAT4(As[load_a_row][load_a_col]) = FLOAT4(a[(by * BM + load_a_row) * k + bk + load_a_col]);
        FLOAT4(As[load_a_row + 64][load_a_col]) = FLOAT4(a[(by * BM + load_a_row + 64) * k + bk + load_a_col]);

        FLOAT4(Bs[load_b_row][load_b_col]) = FLOAT4(b[(bk + load_b_row) * n + bx * BN + load_b_col]);
        FLOAT4(Bs[load_b_row + 8][load_b_col]) = FLOAT4(b[(bk + load_b_row + 8) * n + bx * BN + load_b_col]);

        __syncthreads();

        // 8×8 outer-product accumulation
#pragma unroll
        for (int i = 0; i < BK; i++) {
            float reg_a[TM], reg_b[TN];

#pragma unroll
            for (int m_idx = 0; m_idx < TM; ++m_idx)
                reg_a[m_idx] = As[c_row + m_idx][i];

            FLOAT4(reg_b[0]) = FLOAT4(Bs[i][c_col]);
            FLOAT4(reg_b[4]) = FLOAT4(Bs[i][c_col + 4]);

#pragma unroll
            for (int m_idx = 0; m_idx < TM; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < TN; ++n_idx) {
                    sum[m_idx][n_idx] += reg_a[m_idx] * reg_b[n_idx];
                }
            }
        }
        __syncthreads();
    }

    // Write back C
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        FLOAT4(c[(by * BM + c_row + i) * n + bx * BN + c_col]) = FLOAT4(sum[i][0]);
        FLOAT4(c[(by * BM + c_row + i) * n + bx * BN + c_col + 4]) = FLOAT4(sum[i][4]);
    }
}

// a block calculate c[128][128], each thread c[8][8]
template <const int BM = 128, const int BN = 128, const int BK = 16, const int TM = 8, const int TN = 8>
__global__ void sgemm_at_tiling_kernel(float *a, float *b, float *c, int m, int n, int k) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x; // 0~255; 8 warps, 2x4 warp tiling; each warp 8x4 thread tiling
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // Each block loads 64x16 of A and 8x128 of B per pass, two passes total
    // Every 4 threads load one row of A (16 elems); every 32 threads load one row of B (128 elems)
    int load_a_row = tid / 4;               // 0~63
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12...
    int load_b_row = tid / WARP_SIZE;       // 0~8
    int load_b_col = (tid % WARP_SIZE) * 4; // 0,4,8,12,16,20,24,28...

    // warp tiling
    int warp_row = warp_id / 4;      // 0, 1
    int warp_col = warp_id % 4;      // 0, 1, 2, 3
    int t_row_in_warp = lane_id / 4; // 0~7
    int t_col_in_warp = lane_id % 4; // 0~3

    // C output origin: each thread computes an 8x8 tile, 256 threads × 64 = 128×128
    int c_row = warp_row * 64 + t_row_in_warp * 8;
    int c_col = warp_col * 32 + t_col_in_warp * 8;

    __shared__ float As_T[BK][BM];
    __shared__ float Bs[BK][BN];

    float sum[TM][TN] = {0.f};

    for (int bk = 0; bk < k; bk += BK) {
        // Transpose A into shared memory
        float4 tmp_a0 = FLOAT4(a[(by * BM + load_a_row) * k + bk + load_a_col]);
        As_T[load_a_col + 0][load_a_row] = tmp_a0.x;
        As_T[load_a_col + 1][load_a_row] = tmp_a0.y;
        As_T[load_a_col + 2][load_a_row] = tmp_a0.z;
        As_T[load_a_col + 3][load_a_row] = tmp_a0.w;

        float4 tmp_a1 = FLOAT4(a[(by * BM + load_a_row + 64) * k + bk + load_a_col]);
        As_T[load_a_col + 0][load_a_row + 64] = tmp_a1.x;
        As_T[load_a_col + 1][load_a_row + 64] = tmp_a1.y;
        As_T[load_a_col + 2][load_a_row + 64] = tmp_a1.z;
        As_T[load_a_col + 3][load_a_row + 64] = tmp_a1.w;

        FLOAT4(Bs[load_b_row][load_b_col]) = FLOAT4(b[(bk + load_b_row) * n + bx * BN + load_b_col]);
        FLOAT4(Bs[load_b_row + 8][load_b_col]) = FLOAT4(b[(bk + load_b_row + 8) * n + bx * BN + load_b_col]);

        __syncthreads();

        // 8×8 outer-product accumulation
#pragma unroll
        for (int i = 0; i < BK; i++) {
            float reg_a[TM], reg_b[TN];

            FLOAT4(reg_a[0]) = FLOAT4(As_T[i][c_row]);
            FLOAT4(reg_a[4]) = FLOAT4(As_T[i][c_row + 4]);

            FLOAT4(reg_b[0]) = FLOAT4(Bs[i][c_col]);
            FLOAT4(reg_b[4]) = FLOAT4(Bs[i][c_col + 4]);

#pragma unroll
            for (int m_idx = 0; m_idx < TM; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < TN; ++n_idx) {
                    sum[m_idx][n_idx] += reg_a[m_idx] * reg_b[n_idx];
                }
            }
        }
        __syncthreads();
    }

    // Write back C
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        FLOAT4(c[(by * BM + c_row + i) * n + bx * BN + c_col]) = FLOAT4(sum[i][0]);
        FLOAT4(c[(by * BM + c_row + i) * n + bx * BN + c_col + 4]) = FLOAT4(sum[i][4]);
    }
}

// swizzling bcf
template <const int BM = 128, const int BN = 128, const int BK = 16, const int TM = 8, const int TN = 8>
__global__ void sgemm_at_bcf_swizzling_kernel(float *a, float *b, float *c, int m, int n, int k) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x; // 0~255; 8 warps, 2x4 warp tiling; each warp 8x4 thread tiling
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // Each block loads 64x16 of A and 8x128 of B per pass, two passes total
    // Every 4 threads load one row of A (16 elems); every 32 threads load one row of B (128 elems)
    int load_a_row = tid / 4;               // 0~63
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12...
    int load_b_row = tid / WARP_SIZE;       // 0~8
    int load_b_col = (tid % WARP_SIZE) * 4; // 0,4,8,12,16,20,24,28...

    // warp tiling
    int warp_row = warp_id / 4;      // 0, 1
    int warp_col = warp_id % 4;      // 0, 1, 2, 3
    int t_row_in_warp = lane_id / 4; // 0~7
    int t_col_in_warp = lane_id % 4; // 0~3

    // C output origin: each thread computes an 8x8 tile, 256 threads × 64 = 128×128
    int c_row = warp_row * 64 + t_row_in_warp * 8;
    int c_col = warp_col * 32 + t_col_in_warp * 8;

    __shared__ float As_T[BK][BM];
    __shared__ float Bs[BK][BN];

    float sum[TM][TN] = {0.f};

    for (int bk = 0; bk < k; bk += BK) {
        // Transpose A with swizzle into shared memory
        float4 tmp_a0 = FLOAT4(a[(by * BM + load_a_row) * k + bk + load_a_col]);
        As_T[load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row)] = tmp_a0.x;
        As_T[load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row)] = tmp_a0.y;
        As_T[load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row)] = tmp_a0.z;
        As_T[load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row)] = tmp_a0.w;

        float4 tmp_a1 = FLOAT4(a[(by * BM + load_a_row + 64) * k + bk + load_a_col]);
        As_T[load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row + 64)] = tmp_a1.x;
        As_T[load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row + 64)] = tmp_a1.y;
        As_T[load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row + 64)] = tmp_a1.z;
        As_T[load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row + 64)] = tmp_a1.w;

        FLOAT4(Bs[load_b_row][load_b_col]) = FLOAT4(b[(bk + load_b_row) * n + bx * BN + load_b_col]);
        FLOAT4(Bs[load_b_row + 8][load_b_col]) = FLOAT4(b[(bk + load_b_row + 8) * n + bx * BN + load_b_col]);

        __syncthreads();

        // 8×8 outer-product accumulation
#pragma unroll
        for (int i = 0; i < BK; i++) {
            float reg_a[TM], reg_b[TN];

            FLOAT4(reg_a[0]) = FLOAT4(As_T[i][SWIZZLE_A(i, c_row)]);
            FLOAT4(reg_a[4]) = FLOAT4(As_T[i][SWIZZLE_A(i, c_row + 4)]);

            FLOAT4(reg_b[0]) = FLOAT4(Bs[i][c_col]);
            FLOAT4(reg_b[4]) = FLOAT4(Bs[i][c_col + 4]);

#pragma unroll
            for (int m_idx = 0; m_idx < TM; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < TN; ++n_idx) {
                    sum[m_idx][n_idx] += reg_a[m_idx] * reg_b[n_idx];
                }
            }
        }
        __syncthreads();
    }

    // Write back C
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        FLOAT4(c[(by * BM + c_row + i) * n + bx * BN + c_col]) = FLOAT4(sum[i][0]);
        FLOAT4(c[(by * BM + c_row + i) * n + bx * BN + c_col + 4]) = FLOAT4(sum[i][4]);
    }
}

// swizzling bcf + coalesced stg c
template <const int BM = 128, const int BN = 128, const int BK = 16, const int TM = 8, const int TN = 8>
__global__ void sgemm_at_bcf_swizzling_rw_kernel(float *a, float *b, float *c, int m, int n, int k) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x; // 0~255; 8 warps
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // Each block loads 64x16 of A and 8x128 of B per pass, two passes total
    // Every 4 threads load one row of A (16 elems); every 32 threads load one row of B (128 elems)
    int load_a_row = tid / 4;               // 0~63
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12...
    int load_b_row = tid / WARP_SIZE;       // 0~8
    int load_b_col = (tid % WARP_SIZE) * 4; // 0,4,8,12,16,20,24,28...

    // Thread row offset within warp is still 0 or 8
    int t_row_in_warp = (lane_id / 16) * 8;

    // Each warp covers 16 rows; every 16 threads handle 8 rows × 128 cols, split into two float4 passes.
    // E.g., T0 reads/writes cols 0~3 and 64~67 — every 8 consecutive threads produce 32 contiguous
    // floats (128 bytes) on write-back, achieving perfect transaction coalescing.
    int c_row = warp_id * 16 + t_row_in_warp;
    int c_col_base = (lane_id % 16) * 4;
    int c_col_0 = c_col_base;      // 0~3
    int c_col_1 = c_col_base + 64; // 64~67

    __shared__ float As_T[BK][BM];
    __shared__ float Bs[BK][BN];

    float sum[TM][TN] = {0.f};

    for (int bk = 0; bk < k; bk += BK) {
        // Transpose A with swizzle into shared memory
        float4 tmp_a0 = FLOAT4(a[(by * BM + load_a_row) * k + bk + load_a_col]);
        As_T[load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row)] = tmp_a0.x;
        As_T[load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row)] = tmp_a0.y;
        As_T[load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row)] = tmp_a0.z;
        As_T[load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row)] = tmp_a0.w;

        float4 tmp_a1 = FLOAT4(a[(by * BM + load_a_row + 64) * k + bk + load_a_col]);
        As_T[load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row + 64)] = tmp_a1.x;
        As_T[load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row + 64)] = tmp_a1.y;
        As_T[load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row + 64)] = tmp_a1.z;
        As_T[load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row + 64)] = tmp_a1.w;

        FLOAT4(Bs[load_b_row][load_b_col]) = FLOAT4(b[(bk + load_b_row) * n + bx * BN + load_b_col]);
        FLOAT4(Bs[load_b_row + 8][load_b_col]) = FLOAT4(b[(bk + load_b_row + 8) * n + bx * BN + load_b_col]);

        __syncthreads();

        // 8×8 outer-product accumulation
#pragma unroll
        for (int i = 0; i < BK; i++) {
            float reg_a[TM], reg_b[TN];

            FLOAT4(reg_a[0]) = FLOAT4(As_T[i][SWIZZLE_A(i, c_row)]);
            FLOAT4(reg_a[4]) = FLOAT4(As_T[i][SWIZZLE_A(i, c_row + 4)]);

            FLOAT4(reg_b[0]) = FLOAT4(Bs[i][c_col_0]); // read 0~3
            FLOAT4(reg_b[4]) = FLOAT4(Bs[i][c_col_1]); // read 64~67

#pragma unroll
            for (int m_idx = 0; m_idx < TM; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < TN; ++n_idx) {
                    sum[m_idx][n_idx] += reg_a[m_idx] * reg_b[n_idx];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        FLOAT4(c[(by * BM + c_row + i) * n + bx * BN + c_col_0]) = FLOAT4(sum[i][0]);
        FLOAT4(c[(by * BM + c_row + i) * n + bx * BN + c_col_1]) = FLOAT4(sum[i][4]);
    }
}

template <const int BM = 128, const int BN = 128, const int BK = 16, const int TM = 8, const int TN = 8>
__global__ void sgemm_at_bcf_swizzling_dbf_rw_kernel(float *a, float *b, float *c, int m, int n, int k) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x; // 0~255; 8 warps
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // Load mapping
    int load_a_row = tid / 4;               // 0~63
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12...
    int load_b_row = tid / WARP_SIZE;       // 0~8
    int load_b_col = (tid % WARP_SIZE) * 4; // 0,4,8,12,16,20,24,28...

    // C compute/read/write mapping (same as above)
    int t_row_in_warp = (lane_id / 16) * 8;
    int c_row = warp_id * 16 + t_row_in_warp;
    int c_col_base = (lane_id % 16) * 4;
    int c_col_0 = c_col_base; // 0~3
    // int c_col_1 = c_col_base + 64; // 64~67

    // double buffer
    __shared__ float As_T[2][BK][BM];
    __shared__ float Bs[2][BK][BN];

    float sum[TM][TN] = {0.f};

    // Flat pointers into global memory for easy pipeline advancement
    float *a_ptr = a + (by * BM + load_a_row) * k + load_a_col;
    // float *a_ptr_64 = a + (by * BM + load_a_row + 64) * k + load_a_col;
    float *b_ptr = b + load_b_row * n + bx * BN + load_b_col;
    // float *b_ptr_8 = b + (load_b_row + 8) * n + bx * BN + load_b_col;

    // Prefetch first tile (costs 16 extra regs; commented-out ptrs above keep total under 128 to preserve occupancy)
    float4 tmp_a0 = FLOAT4(a_ptr[0]);
    float4 tmp_a1 = FLOAT4(a_ptr[64 * k]);
    float4 tmp_b0 = FLOAT4(b_ptr[0]);
    float4 tmp_b1 = FLOAT4(b_ptr[8 * n]);

    As_T[0][load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row)] = tmp_a0.x;
    As_T[0][load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row)] = tmp_a0.y;
    As_T[0][load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row)] = tmp_a0.z;
    As_T[0][load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row)] = tmp_a0.w;

    As_T[0][load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row + 64)] = tmp_a1.x;
    As_T[0][load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row + 64)] = tmp_a1.y;
    As_T[0][load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row + 64)] = tmp_a1.z;
    As_T[0][load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row + 64)] = tmp_a1.w;

    FLOAT4(Bs[0][load_b_row][load_b_col]) = tmp_b0;
    FLOAT4(Bs[0][load_b_row + 8][load_b_col]) = tmp_b1;

    __syncthreads();

    // Double buffer indices
    int write_idx = 1;
    int read_idx = 0;
    // Main loop
    for (int bk = BK; bk < k; bk += BK) {
        // Advance pointers along K dimension
        a_ptr += BK;
        b_ptr += BK * n;

        // Prefetch next tile — after issuing LDG, we immediately compute on the current tile
        tmp_a0 = FLOAT4(a_ptr[0]);
        tmp_a1 = FLOAT4(a_ptr[64 * k]);
        tmp_b0 = FLOAT4(b_ptr[0]);
        tmp_b1 = FLOAT4(b_ptr[8 * n]);

        // Compute logic identical to non-pipelined version
#pragma unroll
        for (int i = 0; i < BK; i++) {
            float reg_a[TM], reg_b[TN];

            FLOAT4(reg_a[0]) = FLOAT4(As_T[read_idx][i][SWIZZLE_A(i, c_row)]);
            FLOAT4(reg_a[4]) = FLOAT4(As_T[read_idx][i][SWIZZLE_A(i, c_row + 4)]);

            FLOAT4(reg_b[0]) = FLOAT4(Bs[read_idx][i][c_col_0]);
            FLOAT4(reg_b[4]) = FLOAT4(Bs[read_idx][i][c_col_0 + 64]);

#pragma unroll
            for (int m_idx = 0; m_idx < TM; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < TN; ++n_idx) {
                    sum[m_idx][n_idx] += reg_a[m_idx] * reg_b[n_idx];
                }
            }
        }

        // Computation done — store prefetched registers into SMEM write buffer
        As_T[write_idx][load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row)] = tmp_a0.x;
        As_T[write_idx][load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row)] = tmp_a0.y;
        As_T[write_idx][load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row)] = tmp_a0.z;
        As_T[write_idx][load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row)] = tmp_a0.w;

        As_T[write_idx][load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row + 64)] = tmp_a1.x;
        As_T[write_idx][load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row + 64)] = tmp_a1.y;
        As_T[write_idx][load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row + 64)] = tmp_a1.z;
        As_T[write_idx][load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row + 64)] = tmp_a1.w;

        FLOAT4(Bs[write_idx][load_b_row][load_b_col]) = tmp_b0;
        FLOAT4(Bs[write_idx][load_b_row + 8][load_b_col]) = tmp_b1;

        __syncthreads();
        write_idx ^= 1;
        read_idx ^= 1;
    }
    // Process the last prefetched tile
#pragma unroll
    for (int i = 0; i < BK; i++) {
        float reg_a[TM], reg_b[TN];

        FLOAT4(reg_a[0]) = FLOAT4(As_T[read_idx][i][SWIZZLE_A(i, c_row)]);
        FLOAT4(reg_a[4]) = FLOAT4(As_T[read_idx][i][SWIZZLE_A(i, c_row + 4)]);

        FLOAT4(reg_b[0]) = FLOAT4(Bs[read_idx][i][c_col_0]);
        FLOAT4(reg_b[4]) = FLOAT4(Bs[read_idx][i][c_col_0 + 64]);

#pragma unroll
        for (int m_idx = 0; m_idx < TM; ++m_idx) {
#pragma unroll
            for (int n_idx = 0; n_idx < TN; ++n_idx) {
                sum[m_idx][n_idx] += reg_a[m_idx] * reg_b[n_idx];
            }
        }
    }
    // Pipeline done — write back C
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        FLOAT4(c[(by * BM + c_row + i) * n + bx * BN + c_col_0]) = FLOAT4(sum[i][0]);
        FLOAT4(c[(by * BM + c_row + i) * n + bx * BN + c_col_0 + 64]) = FLOAT4(sum[i][4]);
    }
}

#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

#define binding_func_gen(name, num, element_dtype)                                                                     \
    void name(torch::Tensor a, torch::Tensor b, torch::Tensor c) {                                                     \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        CHECK_T(c);                                                                                                    \
        const int M = a.size(0);                                                                                       \
        const int K = a.size(1);                                                                                       \
        const int N = b.size(1);                                                                                       \
        const dim3 threads_per_block(16, 16);                                                                          \
        const dim3 blocks_per_grid((N + 15) / 16, (M + 15) / 16);                                                      \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
                                                                                                                       \
        name##_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(                                              \
            a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);                                   \
    }

#define binding_tiled_func_gen(name)                                                                                   \
    void name(torch::Tensor a, torch::Tensor b, torch::Tensor c) {                                                     \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        CHECK_T(c);                                                                                                    \
        const int M = a.size(0);                                                                                       \
        const int K = a.size(1);                                                                                       \
        const int N = b.size(1);                                                                                       \
        const int BM = 128;                                                                                            \
        const int BN = 128;                                                                                            \
        const int threads_per_block = 256;                                                                             \
        const dim3 blocks_per_grid((N + BN - 1) / BN, (M + BM - 1) / BM);                                              \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
                                                                                                                       \
        name##_kernel<128, 128, 16, 8, 8><<<blocks_per_grid, threads_per_block, 0, stream>>>(                          \
            a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);                                   \
    }

extern void sgemm_cublas(torch::Tensor a, torch::Tensor b, torch::Tensor c);
extern void sgemm_cublas_tf32(torch::Tensor a, torch::Tensor b, torch::Tensor c);

extern void sgemm_tf32_bt(torch::Tensor a, torch::Tensor b, torch::Tensor c);
extern void sgemm_tf32_bt_swizzle(torch::Tensor a, torch::Tensor b, torch::Tensor c);
extern void sgemm_tf32_bt_swizzle_dbf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
extern void sgemm_tf32_swizzle_bcf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
extern void sgemm_tf32_swizzle_bcf_dbf(torch::Tensor a, torch::Tensor b, torch::Tensor c);

binding_func_gen(sgemm_naive, 1, float);

binding_tiled_func_gen(sgemm_tiling);
binding_tiled_func_gen(sgemm_at_tiling);
binding_tiled_func_gen(sgemm_at_bcf_swizzling);
binding_tiled_func_gen(sgemm_at_bcf_swizzling_rw);
binding_tiled_func_gen(sgemm_at_bcf_swizzling_dbf_rw);

// binding
#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(sgemm_cublas);
    torch_pybinding_func(sgemm_naive);
    torch_pybinding_func(sgemm_tiling);
    torch_pybinding_func(sgemm_at_tiling);
    torch_pybinding_func(sgemm_at_bcf_swizzling);
    torch_pybinding_func(sgemm_at_bcf_swizzling_rw);
    torch_pybinding_func(sgemm_at_bcf_swizzling_dbf_rw);
    // tensor core
    torch_pybinding_func(sgemm_cublas_tf32);
    torch_pybinding_func(sgemm_tf32_bt);
    torch_pybinding_func(sgemm_tf32_bt_swizzle);
    torch_pybinding_func(sgemm_tf32_bt_swizzle_dbf);
    torch_pybinding_func(sgemm_tf32_swizzle_bcf);
    torch_pybinding_func(sgemm_tf32_swizzle_bcf_dbf);
}
