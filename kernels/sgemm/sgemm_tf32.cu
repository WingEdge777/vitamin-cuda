#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/types.h>

#define UINT2(value) (reinterpret_cast<uint2 *>(&(value))[0])
#define FLOAT2(value) (reinterpret_cast<float2 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])

#define SWIZZLE_A(row, col) ((col) ^ (((row >> 1) & 0x3) << 2))

#define SWIZZLE_B(row, col) ((col) ^ (((((((row) >> 1)) ^ ((row) >> 3))) & 0x3) << 2))

#define SWIZZLE_B_F2(row, col) ((col) ^ (((row) & 0x7) << 3))

// ---------------- Inline PTX Assembly Macros ----------------
// cp.async: async copy 16 bytes from gmem (src) to smem (dst_smem_32b)
#define CP_ASYNC_CG(dst_smem_32b, src_global_ptr)                                                                      \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(dst_smem_32b), "l"(src_global_ptr))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP_0() asm volatile("cp.async.wait_group 0;\n" ::)

// ldmatrix
#define LDMATRIX_X4(R0, R1, R2, R3, PTR)                                                                               \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"                                    \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                                              \
                 : "r"(PTR))

#define LDMATRIX_X2(R0, R1, PTR)                                                                                       \
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(R0), "=r"(R1) : "r"(PTR))

// mma.sync
#define M16N8K8(C0, C1, C2, C3, A0, A1, A2, A3, B0, B1)                                                                \
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "                                                 \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"                                         \
                 : "=f"(C0), "=f"(C1), "=f"(C2), "=f"(C3)                                                              \
                 : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1), "f"(C0), "f"(C1), "f"(C2), "f"(C3))

const int WARP_SIZE = 32;

// ------------------------------------------ b trans + ldmatrix  ----------------------------------------------------

// a block calculate c[128][128]
template <const int BM = 128, const int BN = 128, const int BK = 16>
__global__ __launch_bounds__(256, 2) void sgemm_tf32_bt_kernel(float *a, float *b, float *c, int m, int n, int k) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x; // 0~255
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // Load mapping
    int load_a_row = tid / 4;               // 0~63
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12
    int load_b_row = tid / WARP_SIZE;       // 0~7  (K dim)
    int load_b_col = (tid % WARP_SIZE) * 4; // 0~124 (N dim)

    // A stays row-major, B transposed to column-major
    __shared__ float As[BM][BK];
    __shared__ float Bs[BN][BK];

    // 2x4 warp tiling — each warp computes a 64×32 C tile
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // Register budget: 4 M-blocks × 4 N-blocks × 4 regs/block = 64
    float sum[4][4][4] = {0.f};

    // Main loop
    for (int bk = 0; bk < k; bk += BK) {

        // 1. cp.async load A (16-byte aligned)
        uint32_t smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row][load_a_col]));
        uint32_t smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row + 64][load_a_col]));

        float *global_a0 = &a[(by * BM + load_a_row) * k + bk + load_a_col];
        float *global_a1 = &a[(by * BM + load_a_row + 64) * k + bk + load_a_col];

        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);
        CP_ASYNC_COMMIT_GROUP();

        // 2. Load B and manually transpose into SMEM
        float4 tmp_b0 = FLOAT4(b[(bk + load_b_row) * n + bx * BN + load_b_col]);
        float4 tmp_b1 = FLOAT4(b[(bk + load_b_row + 8) * n + bx * BN + load_b_col]);

        // Manually transpose B into SMEM
        Bs[load_b_col + 0][load_b_row] = tmp_b0.x;
        Bs[load_b_col + 1][load_b_row] = tmp_b0.y;
        Bs[load_b_col + 2][load_b_row] = tmp_b0.z;
        Bs[load_b_col + 3][load_b_row] = tmp_b0.w;
        Bs[load_b_col + 0][load_b_row + 8] = tmp_b1.x;
        Bs[load_b_col + 1][load_b_row + 8] = tmp_b1.y;
        Bs[load_b_col + 2][load_b_row + 8] = tmp_b1.z;
        Bs[load_b_col + 3][load_b_row + 8] = tmp_b1.w;

        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();

        // 3. Tensor Core compute (K in 2 steps, consuming 8 K each)
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 8;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // Issue 4 ldmatrix for A fragments (4 × 16 = 64 rows)
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
                // warp_id_m stride is 64
                int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16); // 0~15, 0~15
                int a_col = k_offset + (lane_id / 16) * 4;                // 0,4  8,12
                uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&As[a_row][a_col]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

            // Issue 4 ldmatrix for B fragments (4 × 8 = 32 cols)
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                // warp_id_n stride is 32
                int b_row = warp_id_n * 32 + n_idx * 8 + (lane_id % 8);
                int b_col = k_offset + ((lane_id / 8) % 2) * 4;
                uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[b_row][b_col]));
                LDMATRIX_X2(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
            }

            // MMA core: 4×4 m16n8k8
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < 4; ++n_idx) {
                    M16N8K8(sum[m_idx][n_idx][0],
                            sum[m_idx][n_idx][1],
                            sum[m_idx][n_idx][2],
                            sum[m_idx][n_idx][3],
                            reg_a[m_idx][0],
                            reg_a[m_idx][1],
                            reg_a[m_idx][2],
                            reg_a[m_idx][3],
                            reg_b[n_idx][0],
                            reg_b[n_idx][1]);
                }
            }
        }
        __syncthreads();
    }

    // m16n8k8 C fragment layout mapping
    // c fragments layout:
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688 1688.tf32
    int t_row = lane_id / 4;       // 0~7
    int t_col = (lane_id % 4) * 2; // 0, 2, 4, 6

#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {

            int c_base_row = by * BM + warp_id_m * 64 + m_idx * 16;
            int c_base_col = bx * BN + warp_id_n * 32 + n_idx * 8;

            FLOAT2(c[(c_base_row + t_row) * n + c_base_col + t_col]) = FLOAT2(sum[m_idx][n_idx][0]);
            FLOAT2(c[(c_base_row + t_row + 8) * n + c_base_col + t_col]) = FLOAT2(sum[m_idx][n_idx][2]);
        }
    }
}

template <const int BM = 128, const int BN = 128, const int BK = 16>
__global__ __launch_bounds__(256,
                             2) void sgemm_tf32_bt_swizzle_kernel(float *a, float *b, float *c, int m, int n, int k) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x; // 0~255
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // Load mapping
    int load_a_row = tid / 4;               // 0~63
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12
    int load_b_row = tid / WARP_SIZE;       // 0~7  (K dim)
    int load_b_col = (tid % WARP_SIZE) * 4; // 0~124 (N dim)

    // A stays row-major, B transposed to column-major
    __shared__ float As[BM][BK];
    __shared__ float Bs[BN][BK];

    // Warp tiling: row-major mapping for coalesced global C write-back
    // Each warp computes a 64×32 C tile
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // Register budget: 4 M-blocks × 4 N-blocks × 4 regs/block = 64
    float sum[4][4][4] = {0.f};

    // Main loop
    for (int bk = 0; bk < k; bk += BK) {

        // 1. cp.async load A with swizzling
        uint32_t smem_a0 =
            static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row][SWIZZLE_A(load_a_row, load_a_col)]));
        uint32_t smem_a1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&As[load_a_row + 64][SWIZZLE_A(load_a_row + 64, load_a_col)]));

        float *global_a0 = &a[(by * BM + load_a_row) * k + bk + load_a_col];
        float *global_a1 = &a[(by * BM + load_a_row + 64) * k + bk + load_a_col];

        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);
        CP_ASYNC_COMMIT_GROUP();

        // 2. Load B
        float4 tmp_b0 = FLOAT4(b[(bk + load_b_row) * n + bx * BN + load_b_col]);
        float4 tmp_b1 = FLOAT4(b[(bk + load_b_row + 8) * n + bx * BN + load_b_col]);

        // Scatter-transpose B into SMEM with SWIZZLE_B
        Bs[load_b_col + 0][SWIZZLE_B(load_b_col + 0, load_b_row)] = tmp_b0.x;
        Bs[load_b_col + 1][SWIZZLE_B(load_b_col + 1, load_b_row)] = tmp_b0.y;
        Bs[load_b_col + 2][SWIZZLE_B(load_b_col + 2, load_b_row)] = tmp_b0.z;
        Bs[load_b_col + 3][SWIZZLE_B(load_b_col + 3, load_b_row)] = tmp_b0.w;

        Bs[load_b_col + 0][SWIZZLE_B(load_b_col + 0, load_b_row + 8)] = tmp_b1.x;
        Bs[load_b_col + 1][SWIZZLE_B(load_b_col + 1, load_b_row + 8)] = tmp_b1.y;
        Bs[load_b_col + 2][SWIZZLE_B(load_b_col + 2, load_b_row + 8)] = tmp_b1.z;
        Bs[load_b_col + 3][SWIZZLE_B(load_b_col + 3, load_b_row + 8)] = tmp_b1.w;

        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();

        // 3. Tensor Core compute
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 8;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // Issue 4 ldmatrix for A fragments, decoding swizzled addresses
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
                int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
                int a_col = k_offset + (lane_id / 16) * 4;
                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&As[a_row][SWIZZLE_A(a_row, a_col)]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

            // Issue 4 ldmatrix for B fragments, decoding swizzled addresses
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                int b_row = warp_id_n * 32 + n_idx * 8 + (lane_id % 8);
                int b_col = k_offset + ((lane_id / 8) % 2) * 4;
                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[b_row][SWIZZLE_B(b_row, b_col)]));
                LDMATRIX_X2(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
            }

            // MMA core: 4×4 m16n8k8
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < 4; ++n_idx) {
                    M16N8K8(sum[m_idx][n_idx][0],
                            sum[m_idx][n_idx][1],
                            sum[m_idx][n_idx][2],
                            sum[m_idx][n_idx][3],
                            reg_a[m_idx][0],
                            reg_a[m_idx][1],
                            reg_a[m_idx][2],
                            reg_a[m_idx][3],
                            reg_b[n_idx][0],
                            reg_b[n_idx][1]);
                }
            }
        }
        __syncthreads();
    }

    // ---------------- Write back C ----------------
    int t_row = lane_id / 4;
    int t_col = (lane_id % 4) * 2;

#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int c_base_row = by * BM + warp_id_m * 64 + m_idx * 16;
            int c_base_col = bx * BN + warp_id_n * 32 + n_idx * 8;

            FLOAT2(c[(c_base_row + t_row) * n + c_base_col + t_col]) = FLOAT2(sum[m_idx][n_idx][0]);
            FLOAT2(c[(c_base_row + t_row + 8) * n + c_base_col + t_col]) = FLOAT2(sum[m_idx][n_idx][2]);
        }
    }
}

template <const int BM = 128, const int BN = 128, const int BK = 16>
__global__
__launch_bounds__(256, 2) void sgemm_tf32_bt_swizzle_dbf_kernel(float *a, float *b, float *c, int m, int n, int k) {
    // grid swizzling
    int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8; // grid swizzle tile width

    int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    int by = (linear_id / SWIZZLE_W) % gridDim.y;
    // int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    int load_a_row = tid / 4;
    int load_a_col = (tid % 4) * 4;
    int load_b_row = tid / WARP_SIZE;
    int load_b_col = (tid % WARP_SIZE) * 4;

    // double buffer
    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BN][BK];

    // 2x4 warp tiling
    int warp_id_m = warp_id / 4;
    int warp_id_n = warp_id % 4;

    float sum[4][4][4] = {0.f};
    // ------------------------------------------- Prefetch first tile
    int a_swizzle_col_0 = SWIZZLE_A(load_a_row, load_a_col);
    int a_swizzle_col_1 = SWIZZLE_A(load_a_row + 64, load_a_col);

    uint32_t smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[0][load_a_row][a_swizzle_col_0]));
    uint32_t smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[0][load_a_row + 64][a_swizzle_col_1]));

    float *global_a0 = &a[(by * BM + load_a_row) * k + 0 + load_a_col];
    float *global_a1 = &a[(by * BM + load_a_row + 64) * k + 0 + load_a_col];

    CP_ASYNC_CG(smem_a0, global_a0);
    CP_ASYNC_CG(smem_a1, global_a1);
    CP_ASYNC_COMMIT_GROUP();

    // Synchronously load B into registers
    float4 tmp_b0 = FLOAT4(b[(0 + load_b_row) * n + bx * BN + load_b_col]);
    float4 tmp_b1 = FLOAT4(b[(0 + load_b_row + 8) * n + bx * BN + load_b_col]);

    // Transpose B into SMEM
    Bs[0][load_b_col + 0][SWIZZLE_B(load_b_col + 0, load_b_row)] = tmp_b0.x;
    Bs[0][load_b_col + 1][SWIZZLE_B(load_b_col + 1, load_b_row)] = tmp_b0.y;
    Bs[0][load_b_col + 2][SWIZZLE_B(load_b_col + 2, load_b_row)] = tmp_b0.z;
    Bs[0][load_b_col + 3][SWIZZLE_B(load_b_col + 3, load_b_row)] = tmp_b0.w;

    Bs[0][load_b_col + 0][SWIZZLE_B(load_b_col + 0, load_b_row + 8)] = tmp_b1.x;
    Bs[0][load_b_col + 1][SWIZZLE_B(load_b_col + 1, load_b_row + 8)] = tmp_b1.y;
    Bs[0][load_b_col + 2][SWIZZLE_B(load_b_col + 2, load_b_row + 8)] = tmp_b1.z;
    Bs[0][load_b_col + 3][SWIZZLE_B(load_b_col + 3, load_b_row + 8)] = tmp_b1.w;

    CP_ASYNC_WAIT_GROUP_0();
    __syncthreads();

    int read_idx = 0;
    int write_idx = 1;

    // main loop
    for (int bk = BK; bk < k; bk += BK) {

        // 1. Issue async prefetch of next A tile into write_idx buffer
        smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[write_idx][load_a_row][a_swizzle_col_0]));
        smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[write_idx][load_a_row + 64][a_swizzle_col_1]));
        global_a0 = &a[(by * BM + load_a_row) * k + bk + load_a_col];
        global_a1 = &a[(by * BM + load_a_row + 64) * k + bk + load_a_col];

        // Prefetch next B tile into registers (hides global memory latency)
        tmp_b0 = FLOAT4(b[(bk + load_b_row) * n + bx * BN + load_b_col]);
        tmp_b1 = FLOAT4(b[(bk + load_b_row + 8) * n + bx * BN + load_b_col]);

        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);
        CP_ASYNC_COMMIT_GROUP();

        // 2. Tensor Core compute using current read_idx buffer
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 8;
            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
                int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
                int a_col = k_offset + (lane_id / 16) * 4;
                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&As[read_idx][a_row][SWIZZLE_A(a_row, a_col)]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                int b_row = warp_id_n * 32 + n_idx * 8 + (lane_id % 8);
                int b_col = k_offset + ((lane_id / 8) % 2) * 4;
                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[read_idx][b_row][SWIZZLE_B(b_row, b_col)]));
                LDMATRIX_X2(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
            }

#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < 4; ++n_idx) {
                    M16N8K8(sum[m_idx][n_idx][0],
                            sum[m_idx][n_idx][1],
                            sum[m_idx][n_idx][2],
                            sum[m_idx][n_idx][3],
                            reg_a[m_idx][0],
                            reg_a[m_idx][1],
                            reg_a[m_idx][2],
                            reg_a[m_idx][3],
                            reg_b[n_idx][0],
                            reg_b[n_idx][1]);
                }
            }
        }

        // 3. Store prefetched B registers into SMEM write buffer
        Bs[write_idx][load_b_col + 0][SWIZZLE_B(load_b_col + 0, load_b_row)] = tmp_b0.x;
        Bs[write_idx][load_b_col + 1][SWIZZLE_B(load_b_col + 1, load_b_row)] = tmp_b0.y;
        Bs[write_idx][load_b_col + 2][SWIZZLE_B(load_b_col + 2, load_b_row)] = tmp_b0.z;
        Bs[write_idx][load_b_col + 3][SWIZZLE_B(load_b_col + 3, load_b_row)] = tmp_b0.w;

        Bs[write_idx][load_b_col + 0][SWIZZLE_B(load_b_col + 0, load_b_row + 8)] = tmp_b1.x;
        Bs[write_idx][load_b_col + 1][SWIZZLE_B(load_b_col + 1, load_b_row + 8)] = tmp_b1.y;
        Bs[write_idx][load_b_col + 2][SWIZZLE_B(load_b_col + 2, load_b_row + 8)] = tmp_b1.z;
        Bs[write_idx][load_b_col + 3][SWIZZLE_B(load_b_col + 3, load_b_row + 8)] = tmp_b1.w;

        // 4. Sync
        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();

        // Swap buffers
        read_idx ^= 1;
        write_idx ^= 1;
    }
    // Process last prefetched tile
#pragma unroll
    for (int k_step = 0; k_step < 2; ++k_step) {
        int k_offset = k_step * 8;
        uint32_t reg_a[4][4];
        uint32_t reg_b[4][2];

#pragma unroll
        for (int m_idx = 0; m_idx < 4; ++m_idx) {
            int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
            int a_col = k_offset + (lane_id / 16) * 4;
            uint32_t smem_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&As[read_idx][a_row][SWIZZLE_A(a_row, a_col)]));
            LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
        }

#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int b_row = warp_id_n * 32 + n_idx * 8 + (lane_id % 8);
            int b_col = k_offset + ((lane_id / 8) % 2) * 4;
            uint32_t smem_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[read_idx][b_row][SWIZZLE_B(b_row, b_col)]));
            LDMATRIX_X2(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
        }

#pragma unroll
        for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                M16N8K8(sum[m_idx][n_idx][0],
                        sum[m_idx][n_idx][1],
                        sum[m_idx][n_idx][2],
                        sum[m_idx][n_idx][3],
                        reg_a[m_idx][0],
                        reg_a[m_idx][1],
                        reg_a[m_idx][2],
                        reg_a[m_idx][3],
                        reg_b[n_idx][0],
                        reg_b[n_idx][1]);
            }
        }
    }

    // ---------------- Write back C ----------------
    int t_row = lane_id / 4;
    int t_col = (lane_id % 4) * 2;

#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int c_base_row = by * BM + warp_id_m * 64 + m_idx * 16;
            int c_base_col = bx * BN + warp_id_n * 32 + n_idx * 8;

            FLOAT2(c[(c_base_row + t_row) * n + c_base_col + t_col]) = FLOAT2(sum[m_idx][n_idx][0]);
            FLOAT2(c[(c_base_row + t_row + 8) * n + c_base_col + t_col]) = FLOAT2(sum[m_idx][n_idx][2]);
        }
    }
}

// ------------------------------------ a/b cp.async + shared load Bs ---------------------------------------

// a block calculate c[128][128]
template <const int BM = 128, const int BN = 128, const int BK = 16>
__global__ __launch_bounds__(256,
                             2) void sgemm_tf32_swizzle_bcf_kernel(float *a, float *b, float *c, int m, int n, int k) {
    // grid swizzling
    int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8; // grid swizzle tile width

    int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    int by = (linear_id / SWIZZLE_W) % gridDim.y;
    // int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x; // 0~255
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // Load mapping
    int load_a_row = tid / 4;               // 0~63
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12
    int load_b_row = tid / WARP_SIZE;       // 0~7  (K dim)
    int load_b_col = (tid % WARP_SIZE) * 4; // 0~124 (N dim)

    // A/B both row-major [BM][BK]
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // Warp tiling — each warp computes a 64×32 C tile
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // Register budget: 4 M-blocks × 4 N-blocks × 4 regs/block = 64
    float sum[4][4][4] = {0.f};

    // Main loop
    for (int bk = 0; bk < k; bk += BK) {

        // 1. cp.async load A
        uint32_t smem_a0 =
            static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row][SWIZZLE_A(load_a_row, load_a_col)]));
        uint32_t smem_a1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&As[load_a_row + 64][SWIZZLE_A(load_a_row + 64, load_a_col)]));

        float *global_a0 = &a[(by * BM + load_a_row) * k + bk + load_a_col];
        float *global_a1 = &a[(by * BM + load_a_row + 64) * k + bk + load_a_col];

        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);

        // cp.async load B
        uint32_t smem_b0 =
            static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[load_b_row][SWIZZLE_B_F2(load_b_row, load_b_col)]));
        uint32_t smem_b1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&Bs[load_b_row + 8][SWIZZLE_B_F2(load_b_row + 8, load_b_col)]));

        float *global_b0 = &b[(bk + load_b_row) * n + bx * BN + load_b_col];
        float *global_b1 = &b[(bk + load_b_row + 8) * n + bx * BN + load_b_col];

        CP_ASYNC_CG(smem_b0, global_b0);
        CP_ASYNC_CG(smem_b1, global_b1);
        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();

        // 3. Tensor Core compute (K in 2 steps, consuming 8 K each)
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 8;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // Issue 4 ldmatrix for A fragments (4 × 16 = 64 rows)
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
                int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
                int a_col = k_offset + (lane_id / 16) * 4;
                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&As[a_row][SWIZZLE_A(a_row, a_col)]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

            // B fragments: each thread holds 2 values from rows 0 and 4; every 4 threads cover 8 rows × 1 col
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                int n_base = warp_id_n * 32 + n_idx * 8;
                int b_col = n_base + (lane_id / 4);
                int b_row_0 = k_offset + (lane_id % 4);
                int b_row_1 = k_offset + (lane_id % 4) + 4;

                reg_b[n_idx][0] = __float_as_uint(Bs[b_row_0][SWIZZLE_B_F2(b_row_0, b_col)]);
                reg_b[n_idx][1] = __float_as_uint(Bs[b_row_1][SWIZZLE_B_F2(b_row_1, b_col)]);
            }

            // MMA core: 4×4 m16n8k8
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < 4; ++n_idx) {
                    M16N8K8(sum[m_idx][n_idx][0],
                            sum[m_idx][n_idx][1],
                            sum[m_idx][n_idx][2],
                            sum[m_idx][n_idx][3],
                            reg_a[m_idx][0],
                            reg_a[m_idx][1],
                            reg_a[m_idx][2],
                            reg_a[m_idx][3],
                            reg_b[n_idx][0],
                            reg_b[n_idx][1]);
                }
            }
        }
        __syncthreads();
    }

    // ---------------- Write back C ----------------
    int t_row = lane_id / 4;       // 0~7
    int t_col = (lane_id % 4) * 2; // 0, 2, 4, 6

#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int c_base_row = by * BM + warp_id_m * 64 + m_idx * 16;
            int c_base_col = bx * BN + warp_id_n * 32 + n_idx * 8;

            FLOAT2(c[(c_base_row + t_row) * n + c_base_col + t_col]) = FLOAT2(sum[m_idx][n_idx][0]);
            FLOAT2(c[(c_base_row + t_row + 8) * n + c_base_col + t_col]) = FLOAT2(sum[m_idx][n_idx][2]);
        }
    }
}

// a block calculate c[128][128]
template <const int BM = 128, const int BN = 128, const int BK = 16>
__global__ void sgemm_tf32_swizzle_bcf_dbf_kernel(float *a, float *b, float *c, int m, int n, int k) {
    // grid swizzling
    int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8; // grid swizzle tile width

    int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    int by = (linear_id / SWIZZLE_W) % gridDim.y;

    // int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    int load_a_row = tid / 4;
    int load_a_col = (tid % 4) * 4;
    int load_b_row = tid / WARP_SIZE;
    int load_b_col = (tid % WARP_SIZE) * 4;

    // double buffer
    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BK][BM];

    // 2x4 warp tiling
    int warp_id_m = warp_id / 4;
    int warp_id_n = warp_id % 4;

    float sum[4][4][4] = {0.f};
    // ------------------------------------------- Prefetch first tile
    int a_swizzle_col_0 = SWIZZLE_A(load_a_row, load_a_col);
    int a_swizzle_col_1 = SWIZZLE_A(load_a_row + 64, load_a_col);

    uint32_t smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[0][load_a_row][a_swizzle_col_0]));
    uint32_t smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[0][load_a_row + 64][a_swizzle_col_1]));

    float *global_a0 = &a[(by * BM + load_a_row) * k + 0 + load_a_col];
    float *global_a1 = &a[(by * BM + load_a_row + 64) * k + 0 + load_a_col];
    CP_ASYNC_CG(smem_a0, global_a0);
    CP_ASYNC_CG(smem_a1, global_a1);

    uint32_t smem_b0 =
        static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[0][load_b_row][SWIZZLE_B_F2(load_b_row, load_b_col)]));
    uint32_t smem_b1 = static_cast<uint32_t>(
        __cvta_generic_to_shared(&Bs[0][load_b_row + 8][SWIZZLE_B_F2(load_b_row + 8, load_b_col)]));

    float *global_b0 = &b[(0 + load_b_row) * n + bx * BN + load_b_col];
    float *global_b1 = &b[(0 + load_b_row + 8) * n + bx * BN + load_b_col];

    CP_ASYNC_CG(smem_b0, global_b0);
    CP_ASYNC_CG(smem_b1, global_b1);

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP_0();
    __syncthreads();

    int read_idx = 0;
    int write_idx = 1;

    // main loop
    for (int bk = BK; bk < k; bk += BK) {

        // 1. Issue async prefetch of next A tile into write_idx buffer
        smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[write_idx][load_a_row][a_swizzle_col_0]));
        smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[write_idx][load_a_row + 64][a_swizzle_col_1]));
        global_a0 = &a[(by * BM + load_a_row) * k + bk + load_a_col];
        global_a1 = &a[(by * BM + load_a_row + 64) * k + bk + load_a_col];
        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);

        // cp.async load B
        uint32_t smem_b0 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&Bs[write_idx][load_b_row][SWIZZLE_B_F2(load_b_row, load_b_col)]));
        uint32_t smem_b1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&Bs[write_idx][load_b_row + 8][SWIZZLE_B_F2(load_b_row + 8, load_b_col)]));

        float *global_b0 = &b[(bk + load_b_row) * n + bx * BN + load_b_col];
        float *global_b1 = &b[(bk + load_b_row + 8) * n + bx * BN + load_b_col];

        CP_ASYNC_CG(smem_b0, global_b0);
        CP_ASYNC_CG(smem_b1, global_b1);
        CP_ASYNC_COMMIT_GROUP();

        // 2. Tensor Core compute using current read_idx buffer
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 8;
            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // 4×16: 64 rows in M dimension
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
                int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
                int a_col = k_offset + (lane_id / 16) * 4;
                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&As[read_idx][a_row][SWIZZLE_A(a_row, a_col)]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                int n_base = warp_id_n * 32 + n_idx * 8;
                int b_col = n_base + (lane_id / 4);
                int b_row_0 = k_offset + (lane_id % 4);
                int b_row_1 = k_offset + (lane_id % 4) + 4;

                reg_b[n_idx][0] = __float_as_uint(Bs[read_idx][b_row_0][SWIZZLE_B_F2(b_row_0, b_col)]);
                reg_b[n_idx][1] = __float_as_uint(Bs[read_idx][b_row_1][SWIZZLE_B_F2(b_row_1, b_col)]);
            }

#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < 4; ++n_idx) {
                    M16N8K8(sum[m_idx][n_idx][0],
                            sum[m_idx][n_idx][1],
                            sum[m_idx][n_idx][2],
                            sum[m_idx][n_idx][3],
                            reg_a[m_idx][0],
                            reg_a[m_idx][1],
                            reg_a[m_idx][2],
                            reg_a[m_idx][3],
                            reg_b[n_idx][0],
                            reg_b[n_idx][1]);
                }
            }
        }

        // 3. cp.async sync
        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();

        // Swap buffers
        read_idx ^= 1;
        write_idx ^= 1;
    }
    // Process last prefetched tile
#pragma unroll
    for (int k_step = 0; k_step < 2; ++k_step) {
        int k_offset = k_step * 8;
        uint32_t reg_a[4][4];
        uint32_t reg_b[4][2];

#pragma unroll
        for (int m_idx = 0; m_idx < 4; ++m_idx) {
            int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
            int a_col = k_offset + (lane_id / 16) * 4;
            uint32_t smem_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&As[read_idx][a_row][SWIZZLE_A(a_row, a_col)]));
            LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
        }

#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int n_base = warp_id_n * 32 + n_idx * 8;
            int b_col = n_base + (lane_id / 4);
            int b_row_0 = k_offset + (lane_id % 4);
            int b_row_1 = k_offset + (lane_id % 4) + 4;

            reg_b[n_idx][0] = __float_as_uint(Bs[read_idx][b_row_0][SWIZZLE_B_F2(b_row_0, b_col)]);
            reg_b[n_idx][1] = __float_as_uint(Bs[read_idx][b_row_1][SWIZZLE_B_F2(b_row_1, b_col)]);
        }

#pragma unroll
        for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                M16N8K8(sum[m_idx][n_idx][0],
                        sum[m_idx][n_idx][1],
                        sum[m_idx][n_idx][2],
                        sum[m_idx][n_idx][3],
                        reg_a[m_idx][0],
                        reg_a[m_idx][1],
                        reg_a[m_idx][2],
                        reg_a[m_idx][3],
                        reg_b[n_idx][0],
                        reg_b[n_idx][1]);
            }
        }
    }

    // ---------------- Write back C ----------------
    int t_row = lane_id / 4;
    int t_col = (lane_id % 4) * 2;

#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int c_base_row = by * BM + warp_id_m * 64 + m_idx * 16;
            int c_base_col = bx * BN + warp_id_n * 32 + n_idx * 8;

            FLOAT2(c[(c_base_row + t_row) * n + c_base_col + t_col]) = FLOAT2(sum[m_idx][n_idx][0]);
            FLOAT2(c[(c_base_row + t_row + 8) * n + c_base_col + t_col]) = FLOAT2(sum[m_idx][n_idx][2]);
        }
    }
    // Reuse As/Bs for coalesced C write-back (tricky)
    /**
    float (*Cs)[128] = (float (*)[128])(&As[0][0][0]);

    #define SWIZZLE_C(row, col) ((col) ^ (((row) << 3) & 127))

    int t_row = lane_id / 4;
    int t_col = (lane_id % 4) * 2;

    for (int m_idx = 0; m_idx < 4; ++m_idx) {
        __syncthreads();

#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int smem_row = warp_id_m * 16 + t_row;
            int smem_col = warp_id_n * 32 + n_idx * 8 + t_col;

            FLOAT2(Cs[smem_row][SWIZZLE_C(smem_row, smem_col)]) = FLOAT2(sum[m_idx][n_idx][0]);
            FLOAT2(Cs[smem_row + 8][SWIZZLE_C(smem_row + 8, smem_col)]) = FLOAT2(sum[m_idx][n_idx][2]);
        }

        __syncthreads();

        int t_c_row = tid / 32;
        int t_c_col = (tid % 32) * 4;

#pragma unroll
        for (int step = 0; step < 4; ++step) {
            int smem_row = t_c_row + step * 8;
            int smem_col = t_c_col;

            float4 res = FLOAT4(Cs[smem_row][SWIZZLE_C(smem_row, smem_col)]);

            // Recover physical coordinates
            int global_row = by * BM + m_idx * 16 + (smem_row < 16 ? smem_row : 64 + smem_row - 16);
            int global_col = bx * BN + smem_col;

            FLOAT4(c[global_row * n + global_col]) = res;
        }
    }*/
}

#define CHECK_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA tensor")

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
        name##_kernel<128, 128, 16><<<blocks_per_grid, threads_per_block, 0, stream>>>(                                \
            a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);                                   \
    }

binding_tiled_func_gen(sgemm_tf32_bt);
binding_tiled_func_gen(sgemm_tf32_bt_swizzle);
binding_tiled_func_gen(sgemm_tf32_bt_swizzle_dbf);
binding_tiled_func_gen(sgemm_tf32_swizzle_bcf);
binding_tiled_func_gen(sgemm_tf32_swizzle_bcf_dbf);
