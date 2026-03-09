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

// A 矩阵: 写入时跨度小，按行求余
#define SWIZZLE_A(row, col) ((col) ^ (((row >> 1) % 4) << 2))
// B 矩阵: 写入时列跨度大，除以 4 后求余
#define SWIZZLE_B(row, col) ((col) ^ ((((row) >> 2) % 4) << 2))

// ---------------- 内联 PTX 汇编宏定义 ----------------
// cp.async: 从 Global (src) 异步拷贝 16 Bytes 到 Shared (dst_smem_32b)
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

// ----------------------------------- kernels --------------------------------------------------

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

    // A 保持 Row-Major [BM][BK]，B 转为 Col-Major [BN][BK]
    __shared__ float As[BM][BK];
    __shared__ float Bs[BN][BK];

    // warp tiling
    // 每个 warp 负责  64 x 32 的 C 矩阵块
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // 寄存器总量：M维4块 * N维4块 * 每块4个寄存器 = 64 个浮点寄存器/线程 (寄存器总量完美保持不变)
    float sum[4][4][4] = {0.f};

    // 主循环
    for (int bk = 0; bk < k; bk += BK) {

        // 1. 使用 cp.async 加载 A 矩阵 (16 Bytes 对齐)
        uint32_t smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row][load_a_col]));
        uint32_t smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row + 64][load_a_col]));

        float *global_a0 = &a[(by * BM + load_a_row) * k + bk + load_a_col];
        float *global_a1 = &a[(by * BM + load_a_row + 64) * k + bk + load_a_col];

        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);

        // 2. 加载 B 矩阵并手动转置写入 Shared Memory
        float4 tmp_b0 = FLOAT4(b[(bk + load_b_row) * n + bx * BN + load_b_col]);
        float4 tmp_b1 = FLOAT4(b[(bk + load_b_row + 8) * n + bx * BN + load_b_col]);

        // 提交所有的异步拷贝任务，并阻塞等待当前 Group 里的任务全部完成
        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP_0();

        // 将读取的 B 手动打散转置写入 Shared Memory
        Bs[load_b_col + 0][load_b_row] = tmp_b0.x;
        Bs[load_b_col + 1][load_b_row] = tmp_b0.y;
        Bs[load_b_col + 2][load_b_row] = tmp_b0.z;
        Bs[load_b_col + 3][load_b_row] = tmp_b0.w;
        Bs[load_b_col + 0][load_b_row + 8] = tmp_b1.x;
        Bs[load_b_col + 1][load_b_row + 8] = tmp_b1.y;
        Bs[load_b_col + 2][load_b_row + 8] = tmp_b1.z;
        Bs[load_b_col + 3][load_b_row + 8] = tmp_b1.w;

        __syncthreads();

// 3. Tensor Core 计算阶段 (K 维度走 2 步，每次消耗 8 个 K)
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 8;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

// 发射 4 次 ldmatrix 获取 A 矩阵块 (4 * 16 = 64 行)
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                // warp_id_m 跨度是 64
                int a_row = warp_id_m * 64 + m * 16 + (lane_id % 16);
                int a_col = k_offset + (lane_id / 16) * 4;
                uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&As[a_row][a_col]));
                LDMATRIX_X4(reg_a[m][0], reg_a[m][1], reg_a[m][2], reg_a[m][3], smem_addr);
            }

// 发射 4 次 ldmatrix 获取 B 矩阵块 (4 * 8 = 32 列)
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                // warp_id_n 现在跨度是 32
                int b_row = warp_id_n * 32 + n_idx * 8 + (lane_id % 8);
                int b_col = k_offset + ((lane_id / 8) % 2) * 4;
                uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[b_row][b_col]));
                LDMATRIX_X2(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
            }

// MMA 核心运算：4x4 的黄金比例循环
#pragma unroll
            for (int m = 0; m < 4; ++m) {
#pragma unroll
                for (int n_idx = 0; n_idx < 4; ++n_idx) {
                    M16N8K8(sum[m][n_idx][0],
                            sum[m][n_idx][1],
                            sum[m][n_idx][2],
                            sum[m][n_idx][3],
                            reg_a[m][0],
                            reg_a[m][1],
                            reg_a[m][2],
                            reg_a[m][3],
                            reg_b[n_idx][0],
                            reg_b[n_idx][1]);
                }
            }
        }
        __syncthreads();
    }

    // ================== 核心修改区 3：写回偏移坐标适配 ==================
    // Tensor Core m16n8k8 C 寄存器碎片的标准排布映射法则
    int t_row = lane_id / 4;       // 0~7
    int t_col = (lane_id % 4) * 2; // 0, 2, 4, 6

#pragma unroll
    for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            // 根据新的 Warp 跨度重新计算 Global C 的基址
            int c_base_row = by * BM + warp_id_m * 64 + m * 16;
            int c_base_col = bx * BN + warp_id_n * 32 + n_idx * 8;

            c[(c_base_row + t_row) * n + c_base_col + t_col] = sum[m][n_idx][0];
            c[(c_base_row + t_row) * n + c_base_col + t_col + 1] = sum[m][n_idx][1];
            c[(c_base_row + t_row + 8) * n + c_base_col + t_col] = sum[m][n_idx][2];
            c[(c_base_row + t_row + 8) * n + c_base_col + t_col + 1] = sum[m][n_idx][3];
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

    // 彻底消灭 PAD！A 保持 Row-Major，B 保持 Col-Major
    __shared__ float As[BM][BK];
    __shared__ float Bs[BN][BK];

    // warp tiling：连续的行优先映射，合并全局内存写回访存
    // 每个 warp 负责 64 x 32 的 C 矩阵块
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // 寄存器总量：M维4块 * N维4块 * 每块4个寄存器 = 64 个浮点寄存器/线程
    float sum[4][4][4] = {0.f};

    // 主循环
    for (int bk = 0; bk < k; bk += BK) {

        // 1. 使用 cp.async 加载 A 矩阵，带上 SWIZZLE_A
        int a_swizzle_col_0 = SWIZZLE_A(load_a_row, load_a_col);
        int a_swizzle_col_1 = SWIZZLE_A(load_a_row + 64, load_a_col);
        uint32_t smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row][a_swizzle_col_0]));
        uint32_t smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row + 64][a_swizzle_col_1]));

        float *global_a0 = &a[(by * BM + load_a_row) * k + bk + load_a_col];
        float *global_a1 = &a[(by * BM + load_a_row + 64) * k + bk + load_a_col];

        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);

        // 2. 加载 B 矩阵
        float4 tmp_b0 = FLOAT4(b[(bk + load_b_row) * n + bx * BN + load_b_col]);
        float4 tmp_b1 = FLOAT4(b[(bk + load_b_row + 8) * n + bx * BN + load_b_col]);

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP_0();

        // 将读取的 B 手动打散转置写入 Shared Memory，带上 SWIZZLE_B
        Bs[load_b_col + 0][SWIZZLE_B(load_b_col + 0, load_b_row)] = tmp_b0.x;
        Bs[load_b_col + 1][SWIZZLE_B(load_b_col + 1, load_b_row)] = tmp_b0.y;
        Bs[load_b_col + 2][SWIZZLE_B(load_b_col + 2, load_b_row)] = tmp_b0.z;
        Bs[load_b_col + 3][SWIZZLE_B(load_b_col + 3, load_b_row)] = tmp_b0.w;

        Bs[load_b_col + 0][SWIZZLE_B(load_b_col + 0, load_b_row + 8)] = tmp_b1.x;
        Bs[load_b_col + 1][SWIZZLE_B(load_b_col + 1, load_b_row + 8)] = tmp_b1.y;
        Bs[load_b_col + 2][SWIZZLE_B(load_b_col + 2, load_b_row + 8)] = tmp_b1.z;
        Bs[load_b_col + 3][SWIZZLE_B(load_b_col + 3, load_b_row + 8)] = tmp_b1.w;

        __syncthreads();

// 3. Tensor Core 计算阶段
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 8;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

// 发射 4 次 ldmatrix 获取 A 矩阵块，利用 SWIZZLE_A 解码地址
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                int a_row = warp_id_m * 64 + m * 16 + (lane_id % 16);
                int a_col = k_offset + (lane_id / 16) * 4;
                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&As[a_row][SWIZZLE_A(a_row, a_col)]));
                LDMATRIX_X4(reg_a[m][0], reg_a[m][1], reg_a[m][2], reg_a[m][3], smem_addr);
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

// MMA 核心运算：4x4 黄金正方形
#pragma unroll
            for (int m = 0; m < 4; ++m) {
#pragma unroll
                for (int n_idx = 0; n_idx < 4; ++n_idx) {
                    M16N8K8(sum[m][n_idx][0],
                            sum[m][n_idx][1],
                            sum[m][n_idx][2],
                            sum[m][n_idx][3],
                            reg_a[m][0],
                            reg_a[m][1],
                            reg_a[m][2],
                            reg_a[m][3],
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
    for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int c_base_row = by * BM + warp_id_m * 64 + m * 16;
            int c_base_col = bx * BN + warp_id_n * 32 + n_idx * 8;

            c[(c_base_row + t_row) * n + c_base_col + t_col] = sum[m][n_idx][0];
            c[(c_base_row + t_row) * n + c_base_col + t_col + 1] = sum[m][n_idx][1];
            c[(c_base_row + t_row + 8) * n + c_base_col + t_col] = sum[m][n_idx][2];
            c[(c_base_row + t_row + 8) * n + c_base_col + t_col + 1] = sum[m][n_idx][3];
        }
    }
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
