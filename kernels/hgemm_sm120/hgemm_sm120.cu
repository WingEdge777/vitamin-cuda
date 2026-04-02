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

#define SWIZZLE_A(row, col) ((col) ^ (((row >> 1) & 0x3) << 3))

#define SWIZZLE_B(row, col) ((col) ^ (((row) & 0x7) << 3))

#define SWIZZLE_C(row, col) ((col) ^ (((row) & 0x7) << 3))

// ---------------- 内联 PTX 汇编宏定义 ----------------
// cp.async: 从 gmem (src) 异步拷贝 16 bytes 到 smem (dst_smem_32b)
#define CP_ASYNC_CG(dst_smem_32b, src_global_ptr)                                                                      \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(dst_smem_32b), "l"(src_global_ptr))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    if constexpr (N == 0)
        asm volatile("cp.async.wait_group 0;\n" ::);
    else if constexpr (N == 1)
        asm volatile("cp.async.wait_group 1;\n" ::);
    else if constexpr (N == 2)
        asm volatile("cp.async.wait_group 2;\n" ::);
    else if constexpr (N == 3)
        asm volatile("cp.async.wait_group 3;\n" ::);
}

// ldmatrix
#define LDMATRIX_X4(R0, R1, R2, R3, PTR)                                                                               \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"                                    \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                                              \
                 : "r"(PTR))

#define LDMATRIX_X2(R0, R1, PTR)                                                                                       \
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(R0), "=r"(R1) : "r"(PTR))

// 加载 2 个 8x8 并转置
#define LDMATRIX_X2_TRANS(R0, R1, PTR)                                                                                 \
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];" : "=r"(R0), "=r"(R1) : "r"(PTR))

// mma.sync
#define M16N8K16_F16(C0, C1, C2, C3, A0, A1, A2, A3, B0, B1)                                                           \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "                                                  \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"                                         \
                 : "=f"(C0), "=f"(C1), "=f"(C2), "=f"(C3)                                                              \
                 : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1), "f"(C0), "f"(C1), "f"(C2), "f"(C3))

#define M16N8K16_BF16(C0, C1, C2, C3, A0, A1, A2, A3, B0, B1)                                                          \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "                                                \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"                                         \
                 : "=f"(C0), "=f"(C1), "=f"(C2), "=f"(C3)                                                              \
                 : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1), "f"(C0), "f"(C1), "f"(C2), "f"(C3))

const int WARP_SIZE = 32;

// ------------------------------------------ cp.async + mma  ----------------------------------------------------

// a block calculate c[128][128]
template <const int BM = 128, const int BN = 128, const int BK = 32, typename T>
__global__ void hgemm_bcf_dbf_rw_kernel(T *a, T *b, T *c, int m, int n, int k) {
    // grid swizzling
    int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8; // 将执行块设置为 8 的宽度

    int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    int by = (linear_id / SWIZZLE_W) % gridDim.y;

    int tid = threadIdx.x; // 0~255
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // 搬运映射
    int load_a_row = tid / 4;        // 0~63
    int load_a_col = (tid % 4) * 8;  // 0,8,16,24
    int load_b_row = tid / 16;       // 0~15 (K维度)
    int load_b_col = (tid % 16) * 8; // 0,8,16 ... 120 (N维度)

    // A/B 都行优先,用union复用同一块内存，写法优雅
    __shared__ __align__(128) union {
        // 前半段计算用的 A 和 B
        struct {
            T As[2][BM][BK];
            T Bs[2][BK][BN];
        };
        // 后半段写回用的 C
        T Cs[BM][BN];
    } smem;

    // warp tiling
    // 每个 warp 负责  64 x 32 的 C 矩阵块
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // 寄存器总量：M维4块 * N维4块 * 每块4个寄存器 = 64
    float sum[4][4][4] = {0.f};

    // ----------------------------- Prologue 先加载一次As/Bs
    // cp.async load A
    uint32_t smem_a0 =
        static_cast<uint32_t>(__cvta_generic_to_shared(&smem.As[0][load_a_row][SWIZZLE_A(load_a_row, load_a_col)]));
    uint32_t smem_a1 = static_cast<uint32_t>(
        __cvta_generic_to_shared(&smem.As[0][load_a_row + 64][SWIZZLE_A(load_a_row + 64, load_a_col)]));

    T *global_a0 = &a[(by * BM + load_a_row) * k + load_a_col];
    T *global_a1 = &a[(by * BM + load_a_row + 64) * k + load_a_col];

    CP_ASYNC_CG(smem_a0, global_a0);
    CP_ASYNC_CG(smem_a1, global_a1);

    // cp.async load B
    uint32_t smem_b0 =
        static_cast<uint32_t>(__cvta_generic_to_shared(&smem.Bs[0][load_b_row][SWIZZLE_B(load_b_row, load_b_col)]));
    uint32_t smem_b1 = static_cast<uint32_t>(
        __cvta_generic_to_shared(&smem.Bs[0][load_b_row + 16][SWIZZLE_B(load_b_row + 16, load_b_col)]));

    T *global_b0 = &b[(load_b_row)*n + bx * BN + load_b_col];
    T *global_b1 = &b[(load_b_row + 16) * n + bx * BN + load_b_col];

    CP_ASYNC_CG(smem_b0, global_b0);
    CP_ASYNC_CG(smem_b1, global_b1);

    CP_ASYNC_COMMIT_GROUP();
    cp_async_wait_group<0>();
    __syncthreads();

    int read_idx = 0;
    int write_idx = 1;

    // 主循环
    for (int bk = 32; bk < k; bk += BK) {

        // 1. cp.async load A
        smem_a0 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&smem.As[write_idx][load_a_row][SWIZZLE_A(load_a_row, load_a_col)]));
        smem_a1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&smem.As[write_idx][load_a_row + 64][SWIZZLE_A(load_a_row + 64, load_a_col)]));

        // 全局地址改为跨步，降低寄存器压力
        global_a0 += BK;
        global_a1 += BK;

        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);

        // 2. cp.async load B
        smem_b0 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&smem.Bs[write_idx][load_b_row][SWIZZLE_B(load_b_row, load_b_col)]));
        smem_b1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&smem.Bs[write_idx][load_b_row + 16][SWIZZLE_B(load_b_row + 16, load_b_col)]));

        global_b0 += BK * n;
        global_b1 += BK * n;

        CP_ASYNC_CG(smem_b0, global_b0);
        CP_ASYNC_CG(smem_b1, global_b1);

        CP_ASYNC_COMMIT_GROUP();

        // 3. Tensor Core 计算阶段 (k维度分两次，一次 16 个k)
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 16;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // 4 次 ldmatrix A (4 * 16 = 64 行)
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
                // ldmatrix x4 读 16x16
                int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
                int a_col = k_offset + (lane_id / 16) * 8;
                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&smem.As[read_idx][a_row][SWIZZLE_A(a_row, a_col)]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

            // 4 次 ldmatrix B (4 * 8 = 32 列)
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                // Lane 0~15 的线程读 16 行 (两块 8x8 的首地址)
                int b_row = k_offset + (lane_id % 16);
                int b_col = warp_id_n * 32 + n_idx * 8;

                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&smem.Bs[read_idx][b_row][SWIZZLE_B(b_row, b_col)]));
                LDMATRIX_X2_TRANS(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
            }

            // MMA 核心运算：4x4 次 m16n8k16
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < 4; ++n_idx) {
                    if constexpr (std::is_same_v<T, __half>) {
                        M16N8K16_F16(sum[m_idx][n_idx][0],
                                     sum[m_idx][n_idx][1],
                                     sum[m_idx][n_idx][2],
                                     sum[m_idx][n_idx][3],
                                     reg_a[m_idx][0],
                                     reg_a[m_idx][1],
                                     reg_a[m_idx][2],
                                     reg_a[m_idx][3],
                                     reg_b[n_idx][0],
                                     reg_b[n_idx][1]);
                    } else {
                        M16N8K16_BF16(sum[m_idx][n_idx][0],
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
        }
        cp_async_wait_group<0>();
        __syncthreads();
        read_idx ^= 1;
        write_idx ^= 1;
    }
    // ------------------- Epilogue 最后计算一次再写回
#pragma unroll
    for (int k_step = 0; k_step < 2; ++k_step) {
        int k_offset = k_step * 16;

        uint32_t reg_a[4][4];
        uint32_t reg_b[4][2];

        // 4 次 ldmatrix A (4 * 16 = 64 行)
#pragma unroll
        for (int m_idx = 0; m_idx < 4; ++m_idx) {
            // ldmatrix x4 读 16x16
            int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
            int a_col = k_offset + (lane_id / 16) * 8;
            uint32_t smem_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&smem.As[read_idx][a_row][SWIZZLE_A(a_row, a_col)]));
            LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
        }

        // 4 次 ldmatrix B (4 * 8 = 32 列)
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            // Lane 0~15 的线程读 16 行 (两块 8x8 的首地址)
            int b_row = k_offset + (lane_id % 16);
            int b_col = warp_id_n * 32 + n_idx * 8;

            uint32_t smem_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&smem.Bs[read_idx][b_row][SWIZZLE_B(b_row, b_col)]));
            LDMATRIX_X2_TRANS(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
        }

        // MMA 核心运算：4x4 次 m16n8k16
#pragma unroll
        for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                if constexpr (std::is_same_v<T, __half>) {
                    M16N8K16_F16(sum[m_idx][n_idx][0],
                                 sum[m_idx][n_idx][1],
                                 sum[m_idx][n_idx][2],
                                 sum[m_idx][n_idx][3],
                                 reg_a[m_idx][0],
                                 reg_a[m_idx][1],
                                 reg_a[m_idx][2],
                                 reg_a[m_idx][3],
                                 reg_b[n_idx][0],
                                 reg_b[n_idx][1]);
                } else {
                    M16N8K16_BF16(sum[m_idx][n_idx][0],
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
    }

    // ---------------- 写回 C 矩阵 ----------------
    // 复用 As/Bs 中转
    __syncthreads();

    int t_row = lane_id / 4;       // 0~7
    int t_col = (lane_id % 4) * 2; // 0, 2, 4, 6

    // register to Cs smem
#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int c_base_row = warp_id_m * 64 + m_idx * 16; // m 跨16行
            int c_base_col = warp_id_n * 32 + n_idx * 8;  // n 跨8列

            // 16行我们分成两次8行写入
            int c_row_0 = c_base_row + t_row;
            int c_row_2 = c_base_row + t_row + 8;
            int c_col = c_base_col + t_col;

            if constexpr (std::is_same_v<T, __half>) {
                HALF2(smem.Cs[c_row_0][SWIZZLE_C(c_row_0, c_col)]) = __float22half2_rn(FLOAT2(sum[m_idx][n_idx][0]));
                HALF2(smem.Cs[c_row_2][SWIZZLE_C(c_row_2, c_col)]) = __float22half2_rn(FLOAT2(sum[m_idx][n_idx][2]));
            } else {
                BFLOAT2(smem.Cs[c_row_0][SWIZZLE_C(c_row_0, c_col)]) =
                    __float22bfloat162_rn(FLOAT2(sum[m_idx][n_idx][0]));
                BFLOAT2(smem.Cs[c_row_2][SWIZZLE_C(c_row_2, c_col)]) =
                    __float22bfloat162_rn(FLOAT2(sum[m_idx][n_idx][2]));
            }
        }
    }

    __syncthreads();

    // smem to gmem
    // 每个线程负责搬运 64 个元素 (fp16/bf16)，即 8 个 float4，每次 一个 warp 读写 32*4*4=512B， 256个线程一次写 4096B
    T *c_block = &c[by * BM * n + bx * BN];

#pragma unroll
    for (int step = 0; step < 8; ++step) {
        // 保证同一个 warp 的 32 个线程，此时读取的 elem_idx 是绝对连续的
        int elem_idx = (step * 256 + tid) * 8;
        int row = elem_idx / 128;
        int col = elem_idx % 128;

        int s_col = SWIZZLE_C(row, col);

        FLOAT4(c_block[row * n + col]) = FLOAT4(smem.Cs[row][s_col]);
    }
}

template <const int BM = 128, const int BN = 128, const int BK = 32, const int STAGES = 3, typename T>
__global__ void hgemm_k_stages_kernel(T *a, T *b, T *c, int m, int n, int k) {
    // grid swizzling
    int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8; // 将执行块设置为 8 的宽度

    int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    int by = (linear_id / SWIZZLE_W) % gridDim.y;

    int tid = threadIdx.x; // 0~255
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // 搬运映射
    int load_a_row = tid / 4;        // 0~63
    int load_a_col = (tid % 4) * 8;  // 0,8,16,24
    int load_b_row = tid / 16;       // 0~15 (K维度)
    int load_b_col = (tid % 16) * 8; // 0,8,16 ... 120 (N维度)

    // A/B 都行优先,用union复用同一块内存，写法优雅
    __shared__ __align__(128) union {
        // 前半段计算用的 A 和 B
        struct {
            T As[STAGES][BM][BK];
            T Bs[STAGES][BK][BN];
        };
        // 后半段写回用的 C
        T Cs[BM][BN];
    } smem;

    // warp tiling
    // 每个 warp 负责  64 x 32 的 C 矩阵块
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // 寄存器总量：M维4块 * N维4块 * 每块4个寄存器 = 64
    float sum[4][4][4] = {0.f};

    T *global_a_ptr = &a[(by * BM + load_a_row) * k + load_a_col];
    T *global_b_ptr = &b[load_b_row * n + bx * BN + load_b_col];

    // 1. prologue: 加载stages-1 个As/Bs块
#pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) {
        uint32_t smem_a0 =
            static_cast<uint32_t>(__cvta_generic_to_shared(&smem.As[i][load_a_row][SWIZZLE_A(load_a_row, load_a_col)]));
        uint32_t smem_a1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&smem.As[i][load_a_row + 64][SWIZZLE_A(load_a_row + 64, load_a_col)]));
        // a 矩阵跨 64 行
        CP_ASYNC_CG(smem_a0, global_a_ptr);
        CP_ASYNC_CG(smem_a1, global_a_ptr + 64 * k);

        uint32_t smem_b0 =
            static_cast<uint32_t>(__cvta_generic_to_shared(&smem.Bs[i][load_b_row][SWIZZLE_B(load_b_row, load_b_col)]));
        uint32_t smem_b1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&smem.Bs[i][load_b_row + 16][SWIZZLE_B(load_b_row + 16, load_b_col)]));
        // b 矩阵跨 16 行
        CP_ASYNC_CG(smem_b0, global_b_ptr);
        CP_ASYNC_CG(smem_b1, global_b_ptr + 16 * n);

        CP_ASYNC_COMMIT_GROUP();

        global_a_ptr += BK;
        global_b_ptr += BK * n;
    }
    // commit了两个group, 允许1个group后台在cp.async, 即等最早加载的load完毕
    cp_async_wait_group<STAGES - 2>();
    __syncthreads();

    // 状态指针初始化
    int load_stage = STAGES - 1; // 下一个要 Load 的位置
    int compute_stage = 0;       // 当前要 Compute 的位置

    // 2. main loop
    for (int bk = (STAGES - 1) * BK; bk < k; bk += BK) {

        // 1. 先发起cp.async load As 到 load_stage
        uint32_t smem_a0 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&smem.As[load_stage][load_a_row][SWIZZLE_A(load_a_row, load_a_col)]));
        uint32_t smem_a1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&smem.As[load_stage][load_a_row + 64][SWIZZLE_A(load_a_row + 64, load_a_col)]));
        CP_ASYNC_CG(smem_a0, global_a_ptr);
        CP_ASYNC_CG(smem_a1, global_a_ptr + 64 * k);

        // 2. 先发起cp.async load As 到 load_stage
        uint32_t smem_b0 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&smem.Bs[load_stage][load_b_row][SWIZZLE_B(load_b_row, load_b_col)]));
        uint32_t smem_b1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&smem.Bs[load_stage][load_b_row + 16][SWIZZLE_B(load_b_row + 16, load_b_col)]));
        CP_ASYNC_CG(smem_b0, global_b_ptr);
        CP_ASYNC_CG(smem_b1, global_b_ptr + 16 * n);

        CP_ASYNC_COMMIT_GROUP();

        // 3. Tensor Core 计算阶段 (k维度分两次，一次 16 个k)
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 16;
            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // 4 次 ldmatrix A (4 * 16 = 64 行)
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
                // ldmatrix x4 读 16x16
                int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
                int a_col = k_offset + (lane_id / 16) * 8;
                uint32_t smem_addr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(&smem.As[compute_stage][a_row][SWIZZLE_A(a_row, a_col)]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

            // 4 次 ldmatrix B (4 * 8 = 32 列)
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                // Lane 0~15 的线程读 16 行 (两块 8x8 的首地址)
                int b_row = k_offset + (lane_id % 16);
                int b_col = warp_id_n * 32 + n_idx * 8;

                uint32_t smem_addr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(&smem.Bs[compute_stage][b_row][SWIZZLE_B(b_row, b_col)]));
                LDMATRIX_X2_TRANS(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
            }

            // MMA 核心运算：4x4 次 m16n8k16
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < 4; ++n_idx) {
                    if constexpr (std::is_same_v<T, __half>) {
                        M16N8K16_F16(sum[m_idx][n_idx][0],
                                     sum[m_idx][n_idx][1],
                                     sum[m_idx][n_idx][2],
                                     sum[m_idx][n_idx][3],
                                     reg_a[m_idx][0],
                                     reg_a[m_idx][1],
                                     reg_a[m_idx][2],
                                     reg_a[m_idx][3],
                                     reg_b[n_idx][0],
                                     reg_b[n_idx][1]);
                    } else {
                        M16N8K16_BF16(sum[m_idx][n_idx][0],
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
        }

        // 保障最早的 group load 好
        cp_async_wait_group<STAGES - 2>();
        __syncthreads();

        // 推进指针
        global_a_ptr += BK;
        global_b_ptr += BK * n;

        // 流水线往下推一级
        load_stage = (load_stage + 1 == STAGES) ? 0 : load_stage + 1;
        compute_stage = (compute_stage + 1 == STAGES) ? 0 : compute_stage + 1;
    }

    // 3. epilogue 最后计算 stages-1 次 再写回
    cp_async_wait_group<0>();
    __syncthreads();
#pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) {
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 16;
            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // 4 次 ldmatrix A (4 * 16 = 64 行)
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
                // ldmatrix x4 读 16x16
                int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
                int a_col = k_offset + (lane_id / 16) * 8;
                uint32_t smem_addr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(&smem.As[compute_stage][a_row][SWIZZLE_A(a_row, a_col)]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

            // 4 次 ldmatrix B (4 * 8 = 32 列)
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                // Lane 0~15 的线程读 16 行 (两块 8x8 的首地址)
                int b_row = k_offset + (lane_id % 16);
                int b_col = warp_id_n * 32 + n_idx * 8;

                uint32_t smem_addr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(&smem.Bs[compute_stage][b_row][SWIZZLE_B(b_row, b_col)]));
                LDMATRIX_X2_TRANS(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
            }

            // MMA 核心运算：4x4 次 m16n8k16
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < 4; ++n_idx) {
                    if constexpr (std::is_same_v<T, __half>) {
                        M16N8K16_F16(sum[m_idx][n_idx][0],
                                     sum[m_idx][n_idx][1],
                                     sum[m_idx][n_idx][2],
                                     sum[m_idx][n_idx][3],
                                     reg_a[m_idx][0],
                                     reg_a[m_idx][1],
                                     reg_a[m_idx][2],
                                     reg_a[m_idx][3],
                                     reg_b[n_idx][0],
                                     reg_b[n_idx][1]);
                    } else {
                        M16N8K16_BF16(sum[m_idx][n_idx][0],
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
        }

        compute_stage = (compute_stage + 1 == STAGES) ? 0 : compute_stage + 1;
    }

    // ---------------- 写回 C 矩阵 ----------------
    // 复用 As/Bs 中转
    __syncthreads();

    int t_row = lane_id / 4;       // 0~7
    int t_col = (lane_id % 4) * 2; // 0, 2, 4, 6

    // register to Cs smem
#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int c_base_row = warp_id_m * 64 + m_idx * 16; // m 跨16行
            int c_base_col = warp_id_n * 32 + n_idx * 8;  // n 跨8列

            // 16行我们分成两次8行写入
            int c_row_0 = c_base_row + t_row;
            int c_row_2 = c_base_row + t_row + 8;
            int c_col = c_base_col + t_col;

            if constexpr (std::is_same_v<T, __half>) {
                HALF2(smem.Cs[c_row_0][SWIZZLE_C(c_row_0, c_col)]) = __float22half2_rn(FLOAT2(sum[m_idx][n_idx][0]));
                HALF2(smem.Cs[c_row_2][SWIZZLE_C(c_row_2, c_col)]) = __float22half2_rn(FLOAT2(sum[m_idx][n_idx][2]));
            } else {
                BFLOAT2(smem.Cs[c_row_0][SWIZZLE_C(c_row_0, c_col)]) =
                    __float22bfloat162_rn(FLOAT2(sum[m_idx][n_idx][0]));
                BFLOAT2(smem.Cs[c_row_2][SWIZZLE_C(c_row_2, c_col)]) =
                    __float22bfloat162_rn(FLOAT2(sum[m_idx][n_idx][2]));
            }
        }
    }

    __syncthreads();

    // smem to gmem
    // 每个线程负责搬运 64 个元素 (fp16/bf16)，即 8 个 float4，每次 一个 warp 读写 32*4*4=512B， 256个线程一次写 4096B
    T *c_block = &c[by * BM * n + bx * BN];

#pragma unroll
    for (int step = 0; step < 8; ++step) {
        // 保证同一个 warp 的 32 个线程，此时读取的 elem_idx 是绝对连续的
        int elem_idx = (step * 256 + tid) * 8;
        int row = elem_idx / 128;
        int col = elem_idx % 128;

        int s_col = SWIZZLE_C(row, col);

        FLOAT4(c_block[row * n + col]) = FLOAT4(smem.Cs[row][s_col]);
    }
}
// -------------------   tma r + mma -------------------
// a block calculate c[128][128]
template <const int BM = 128, const int BN = 128, const int BK = 32, const int STAGES = 3, typename T>
__global__ void hgemm_tma_r_k_stages_kernel(T *a, T *b, T *c, int m, int n, int k) {}

// -------------------  tma rw + mma -------------------
// a block calculate c[128][128]
template <const int BM = 128, const int BN = 128, const int BK = 32, const int STAGES = 3, typename T>
__global__ void hgemm_tma_rw_k_stages_kernel(T *a, T *b, T *c, int m, int n, int k) {}

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
        if (a.dtype() == torch::kHalf) {                                                                               \
            name##_kernel<128, 128, 32>                                                                                \
                <<<blocks_per_grid, threads_per_block, 0, stream>>>(reinterpret_cast<__half *>(a.data_ptr()),          \
                                                                    reinterpret_cast<__half *>(b.data_ptr()),          \
                                                                    reinterpret_cast<__half *>(c.data_ptr()),          \
                                                                    M,                                                 \
                                                                    N,                                                 \
                                                                    K);                                                \
        } else {                                                                                                       \
            name##_kernel<128, 128, 32>                                                                                \
                <<<blocks_per_grid, threads_per_block, 0, stream>>>(reinterpret_cast<__nv_bfloat16 *>(a.data_ptr()),   \
                                                                    reinterpret_cast<__nv_bfloat16 *>(b.data_ptr()),   \
                                                                    reinterpret_cast<__nv_bfloat16 *>(c.data_ptr()),   \
                                                                    M,                                                 \
                                                                    N,                                                 \
                                                                    K);                                                \
        }                                                                                                              \
    }

binding_tiled_func_gen(hgemm_bcf_dbf_rw);
binding_tiled_func_gen(hgemm_k_stages);

extern void hgemm_cublas(torch::Tensor a, torch::Tensor b, torch::Tensor c);

// binding
#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(hgemm_cublas);
    torch_pybinding_func(hgemm_bcf_dbf_rw);
    torch_pybinding_func(hgemm_k_stages);
}
