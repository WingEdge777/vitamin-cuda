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

// ---------------- 内联 PTX 汇编宏定义 ----------------
// cp.async: 从 gmem (src) 异步拷贝 16 bytes 到 smem (dst_smem_32b)
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

    // 搬运映射
    int load_a_row = tid / 4;               // 0~63
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12
    int load_b_row = tid / WARP_SIZE;       // 0~7  (K维度)
    int load_b_col = (tid % WARP_SIZE) * 4; // 0~124 (N维度)

    // A 保持 行优先，B 转置为 列优先
    __shared__ float As[BM][BK];
    __shared__ float Bs[BN][BK];

    // 2x4 warp tiling
    // 一行 warp 负责上下64x128， 每个 warp 负责  64 x 32 的 C 矩阵块
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // 寄存器总量：M维4块 * N维4块 * 每块4个寄存器 = 64
    float sum[4][4][4] = {0.f};

    // 主循环
    for (int bk = 0; bk < k; bk += BK) {

        // 1. 使用 cp.async 加载 A 矩阵 (16 bytes 对齐)
        uint32_t smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row][load_a_col]));
        uint32_t smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row + 64][load_a_col]));

        float *global_a0 = &a[(by * BM + load_a_row) * k + bk + load_a_col];
        float *global_a1 = &a[(by * BM + load_a_row + 64) * k + bk + load_a_col];

        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);
        // 提交所有的异步拷贝任务
        CP_ASYNC_COMMIT_GROUP();

        // 2. 加载 B 矩阵并手动转置写入 smem
        float4 tmp_b0 = FLOAT4(b[(bk + load_b_row) * n + bx * BN + load_b_col]);
        float4 tmp_b1 = FLOAT4(b[(bk + load_b_row + 8) * n + bx * BN + load_b_col]);

        // 将读取的 B 手动转置写入 smem
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

        // 3. Tensor Core 计算阶段 (K 维度走 2 步，每次消耗 8 个 K)
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 8;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // 发射 4 次 ldmatrix 获取 A 矩阵块 (4 * 16 = 64 行)
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
                // warp_id_m 跨度是 64
                int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16); // 0~15, 0~15
                int a_col = k_offset + (lane_id / 16) * 4;                // 0,4  8,12
                uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&As[a_row][a_col]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

            // 发射 4 次 ldmatrix 获取 B 矩阵块 (4 * 8 = 32 列)
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                // warp_id_n 跨度是 32
                int b_row = warp_id_n * 32 + n_idx * 8 + (lane_id % 8);
                int b_col = k_offset + ((lane_id / 8) % 2) * 4;
                uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[b_row][b_col]));
                LDMATRIX_X2(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
            }

            // MMA 核心运算：4x4 的 1688
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

    // Tensor Core m16n8k8 C 寄存器碎片的标准排布映射法则
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

            c[(c_base_row + t_row) * n + c_base_col + t_col] = sum[m_idx][n_idx][0];
            c[(c_base_row + t_row) * n + c_base_col + t_col + 1] = sum[m_idx][n_idx][1];
            c[(c_base_row + t_row + 8) * n + c_base_col + t_col] = sum[m_idx][n_idx][2];
            c[(c_base_row + t_row + 8) * n + c_base_col + t_col + 1] = sum[m_idx][n_idx][3];
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

    // 搬运映射
    int load_a_row = tid / 4;               // 0~63
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12
    int load_b_row = tid / WARP_SIZE;       // 0~7  (K维度)
    int load_b_col = (tid % WARP_SIZE) * 4; // 0~124 (N维度)

    // A 保持 行优先，B 转置为 列优先
    __shared__ float As[BM][BK];
    __shared__ float Bs[BN][BK];

    // warp tiling：连续的行优先映射，合并全局内存写回访存
    // 每个 warp 负责 64 x 32 的 C 矩阵块
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // 寄存器总量：M维4块 * N维4块 * 每块4个寄存器 = 64
    float sum[4][4][4] = {0.f};

    // 主循环
    for (int bk = 0; bk < k; bk += BK) {

        // 1. 使用 cp.async swizzling 加载 A 矩阵
        uint32_t smem_a0 =
            static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row][SWIZZLE_A(load_a_row, load_a_col)]));
        uint32_t smem_a1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&As[load_a_row + 64][SWIZZLE_A(load_a_row + 64, load_a_col)]));

        float *global_a0 = &a[(by * BM + load_a_row) * k + bk + load_a_col];
        float *global_a1 = &a[(by * BM + load_a_row + 64) * k + bk + load_a_col];

        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);
        CP_ASYNC_COMMIT_GROUP(); // 提交

        // 2. 加载 B 矩阵
        float4 tmp_b0 = FLOAT4(b[(bk + load_b_row) * n + bx * BN + load_b_col]);
        float4 tmp_b1 = FLOAT4(b[(bk + load_b_row + 8) * n + bx * BN + load_b_col]);

        // 将读取的 B 手动打散转置写入 smem，带上 SWIZZLE_B
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

        // 3. Tensor Core 计算阶段
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 8;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // 发射 4 次 ldmatrix 获取 A 矩阵块，利用 SWIZZLE_A 解码地址
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
                int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
                int a_col = k_offset + (lane_id / 16) * 4;
                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&As[a_row][SWIZZLE_A(a_row, a_col)]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

            // 发射 4 次 ldmatrix 获取 B 矩阵块，利用 SWIZZLE_B 解码地址
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                int b_row = warp_id_n * 32 + n_idx * 8 + (lane_id % 8);
                int b_col = k_offset + ((lane_id / 8) % 2) * 4;
                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[b_row][SWIZZLE_B(b_row, b_col)]));
                LDMATRIX_X2(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
            }

            // MMA 核心运算：4x4 的 1688
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

    // ---------------- 写回 C 矩阵 ----------------
    int t_row = lane_id / 4;
    int t_col = (lane_id % 4) * 2;

#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int c_base_row = by * BM + warp_id_m * 64 + m_idx * 16;
            int c_base_col = bx * BN + warp_id_n * 32 + n_idx * 8;

            c[(c_base_row + t_row) * n + c_base_col + t_col] = sum[m_idx][n_idx][0];
            c[(c_base_row + t_row) * n + c_base_col + t_col + 1] = sum[m_idx][n_idx][1];
            c[(c_base_row + t_row + 8) * n + c_base_col + t_col] = sum[m_idx][n_idx][2];
            c[(c_base_row + t_row + 8) * n + c_base_col + t_col + 1] = sum[m_idx][n_idx][3];
        }
    }
}

template <const int BM = 128, const int BN = 128, const int BK = 16>
__global__
__launch_bounds__(256, 2) void sgemm_tf32_bt_swizzle_dbf_kernel(float *a, float *b, float *c, int m, int n, int k) {
    // grid swizzling
    int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8; // 将执行块设置为 8 的宽度

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
    // ------------------------------------------- 预先加载一块
    int a_swizzle_col_0 = SWIZZLE_A(load_a_row, load_a_col);
    int a_swizzle_col_1 = SWIZZLE_A(load_a_row + 64, load_a_col);

    uint32_t smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[0][load_a_row][a_swizzle_col_0]));
    uint32_t smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[0][load_a_row + 64][a_swizzle_col_1]));

    float *global_a0 = &a[(by * BM + load_a_row) * k + 0 + load_a_col];
    float *global_a1 = &a[(by * BM + load_a_row + 64) * k + 0 + load_a_col];

    CP_ASYNC_CG(smem_a0, global_a0);
    CP_ASYNC_CG(smem_a1, global_a1);
    CP_ASYNC_COMMIT_GROUP();

    // B 矩阵同步加载进物理寄存器
    float4 tmp_b0 = FLOAT4(b[(0 + load_b_row) * n + bx * BN + load_b_col]);
    float4 tmp_b1 = FLOAT4(b[(0 + load_b_row + 8) * n + bx * BN + load_b_col]);

    // B 矩阵转置写入 SMEM
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

        // 1. 发射异步指令，预取【下一轮】的 A 矩阵到 write_idx 缓冲
        smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[write_idx][load_a_row][a_swizzle_col_0]));
        smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[write_idx][load_a_row + 64][a_swizzle_col_1]));
        global_a0 = &a[(by * BM + load_a_row) * k + bk + load_a_col];
        global_a1 = &a[(by * BM + load_a_row + 64) * k + bk + load_a_col];

        // 预取【下一轮】的 B 矩阵到物理寄存器 (掩盖 Global Memory 延迟)
        tmp_b0 = FLOAT4(b[(bk + load_b_row) * n + bx * BN + load_b_col]);
        tmp_b1 = FLOAT4(b[(bk + load_b_row + 8) * n + bx * BN + load_b_col]);

        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);
        CP_ASYNC_COMMIT_GROUP();

        // 2. 使用当前轮的 read_idx 缓冲进行Tensor Core 计算
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

        // 3. 将预取在寄存器里的【下一轮】 B 矩阵，写入 smem
        Bs[write_idx][load_b_col + 0][SWIZZLE_B(load_b_col + 0, load_b_row)] = tmp_b0.x;
        Bs[write_idx][load_b_col + 1][SWIZZLE_B(load_b_col + 1, load_b_row)] = tmp_b0.y;
        Bs[write_idx][load_b_col + 2][SWIZZLE_B(load_b_col + 2, load_b_row)] = tmp_b0.z;
        Bs[write_idx][load_b_col + 3][SWIZZLE_B(load_b_col + 3, load_b_row)] = tmp_b0.w;

        Bs[write_idx][load_b_col + 0][SWIZZLE_B(load_b_col + 0, load_b_row + 8)] = tmp_b1.x;
        Bs[write_idx][load_b_col + 1][SWIZZLE_B(load_b_col + 1, load_b_row + 8)] = tmp_b1.y;
        Bs[write_idx][load_b_col + 2][SWIZZLE_B(load_b_col + 2, load_b_row + 8)] = tmp_b1.z;
        Bs[write_idx][load_b_col + 3][SWIZZLE_B(load_b_col + 3, load_b_row + 8)] = tmp_b1.w;

        // 4. 同步
        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();

        // 切换缓冲
        read_idx ^= 1;
        write_idx ^= 1;
    }
    // 最后load的数据还有一次计算
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

    // ---------------- 写回 C 矩阵 ----------------
    int t_row = lane_id / 4;
    int t_col = (lane_id % 4) * 2;

#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int c_base_row = by * BM + warp_id_m * 64 + m_idx * 16;
            int c_base_col = bx * BN + warp_id_n * 32 + n_idx * 8;

            c[(c_base_row + t_row) * n + c_base_col + t_col] = sum[m_idx][n_idx][0];
            c[(c_base_row + t_row) * n + c_base_col + t_col + 1] = sum[m_idx][n_idx][1];
            c[(c_base_row + t_row + 8) * n + c_base_col + t_col] = sum[m_idx][n_idx][2];
            c[(c_base_row + t_row + 8) * n + c_base_col + t_col + 1] = sum[m_idx][n_idx][3];
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
    const int SWIZZLE_W = 8; // 将执行块设置为 8 的宽度

    int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    int by = (linear_id / SWIZZLE_W) % gridDim.y;
    // int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x; // 0~255
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // 搬运映射
    int load_a_row = tid / 4;               // 0~63
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12
    int load_b_row = tid / WARP_SIZE;       // 0~7  (K维度)
    int load_b_col = (tid % WARP_SIZE) * 4; // 0~124 (N维度)

    // A/B 保持 Row-Major [BM][BK]，
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // warp tiling
    // 每个 warp 负责  64 x 32 的 C 矩阵块
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // 寄存器总量：M维4块 * N维4块 * 每块4个寄存器 = 64
    float sum[4][4][4] = {0.f};

    // 主循环
    for (int bk = 0; bk < k; bk += BK) {

        // 1. 使用 cp.async 加载 A 矩阵
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
        // 提交所有的异步拷贝任务
        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();

        // 3. Tensor Core 计算阶段 (K 维度走 2 步，每次消耗 8 个 K)
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 8;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // 发射 4 次 ldmatrix 获取 A 矩阵块 (4 * 16 = 64 行)
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
                int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
                int a_col = k_offset + (lane_id / 16) * 4;
                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&As[a_row][SWIZZLE_A(a_row, a_col)]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

            // ldmatrix 结果 也就是mma对 b fragments的要求是，一线程两个值，分别在第0、4行，每4线程hold 8行1列的数据
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                // 当前处理的 N 维度的基础列号
                int n_base = warp_id_n * 32 + n_idx * 8;

                // 每四个线程一列
                int b_col = n_base + (lane_id / 4);

                // k维度的 0~3 行 和 4~7 行
                int b_row_0 = k_offset + (lane_id % 4);
                int b_row_1 = k_offset + (lane_id % 4) + 4;

                // swizzling 读取
                reg_b[n_idx][0] = __float_as_uint(Bs[b_row_0][SWIZZLE_B_F2(b_row_0, b_col)]);
                reg_b[n_idx][1] = __float_as_uint(Bs[b_row_1][SWIZZLE_B_F2(b_row_1, b_col)]);
            }

            // MMA 核心运算：4x4 的 1688
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

    // ---------------- 写回 C 矩阵 ----------------
    int t_row = lane_id / 4;       // 0~7
    int t_col = (lane_id % 4) * 2; // 0, 2, 4, 6

#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int c_base_row = by * BM + warp_id_m * 64 + m_idx * 16;
            int c_base_col = bx * BN + warp_id_n * 32 + n_idx * 8;

            c[(c_base_row + t_row) * n + c_base_col + t_col] = sum[m_idx][n_idx][0];
            c[(c_base_row + t_row) * n + c_base_col + t_col + 1] = sum[m_idx][n_idx][1];
            c[(c_base_row + t_row + 8) * n + c_base_col + t_col] = sum[m_idx][n_idx][2];
            c[(c_base_row + t_row + 8) * n + c_base_col + t_col + 1] = sum[m_idx][n_idx][3];
        }
    }
}

// a block calculate c[128][128]
template <const int BM = 128, const int BN = 128, const int BK = 16>
__global__ void sgemm_tf32_swizzle_bcf_dbf_kernel(float *a, float *b, float *c, int m, int n, int k) {
    // grid swizzling
    int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8; // 将执行块设置为 8 的宽度

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
    // ------------------------------------------- 预先加载一块
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

        // 1. 发射异步指令，预取【下一轮】的 A 矩阵到 write_idx 缓冲
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
        // 提交所有的异步拷贝任务
        CP_ASYNC_COMMIT_GROUP();

        // 2. 使用当前轮的 read_idx 缓冲进行Tensor Core 计算
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 8;
            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // 4x16 : m 维度 64 行
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
                // 当前处理的 N 维度的基础列号
                int n_base = warp_id_n * 32 + n_idx * 8;

                // 每四个线程一列
                int b_col = n_base + (lane_id / 4);

                // k维度的 0~3 行 和 4~7 行
                int b_row_0 = k_offset + (lane_id % 4);
                int b_row_1 = k_offset + (lane_id % 4) + 4;

                // swizzling 读取
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

        // 3. cp.async 同步
        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();

        // 切换缓冲
        read_idx ^= 1;
        write_idx ^= 1;
    }
    // 最后load的数据还有一次计算
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
            // 当前处理的 N 维度的基础列号
            int n_base = warp_id_n * 32 + n_idx * 8;

            // 每四个线程一列
            int b_col = n_base + (lane_id / 4);

            // k维度的 0~3 行 和 4~7 行
            int b_row_0 = k_offset + (lane_id % 4);
            int b_row_1 = k_offset + (lane_id % 4) + 4;

            // swizzling 读取
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

    // ---------------- 写回 C 矩阵 ----------------
    int t_row = lane_id / 4;
    int t_col = (lane_id % 4) * 2;

#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int c_base_row = by * BM + warp_id_m * 64 + m_idx * 16;
            int c_base_col = bx * BN + warp_id_n * 32 + n_idx * 8;

            c[(c_base_row + t_row) * n + c_base_col + t_col] = sum[m_idx][n_idx][0];
            c[(c_base_row + t_row) * n + c_base_col + t_col + 1] = sum[m_idx][n_idx][1];
            c[(c_base_row + t_row + 8) * n + c_base_col + t_col] = sum[m_idx][n_idx][2];
            c[(c_base_row + t_row + 8) * n + c_base_col + t_col + 1] = sum[m_idx][n_idx][3];
        }
    }
    // 复用 As，Bs，合并写回c。有点trick
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

            // 还原物理坐标
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
