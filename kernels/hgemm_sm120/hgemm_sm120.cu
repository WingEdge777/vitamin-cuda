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

#define SWIZZLE_128B_TMA(row, col) ((col) ^ (((row >> 1) & 0x7) << 3))

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

// ---------------- MBarrier 与 TMA 纯 PTX 原语 ----------------

// MBarrier 类型定义 (硬件要求 8 字节对齐)
typedef uint64_t mbarrier_t;

// 初始化屏障 (只需在 prologue 中由单线程调用)
__device__ __forceinline__ void mbarrier_init(mbarrier_t *mbar, uint32_t expected_count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar))),
                 "r"(expected_count));
}

// 设定 TMA 传输的预期字节数
__device__ __forceinline__ void mbarrier_expect_tx(mbarrier_t *mbar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" ::"r"(
                     static_cast<uint32_t>(__cvta_generic_to_shared(mbar))),
                 "r"(tx_bytes));
}

// 计算线程消费完数据后，提交到达信号 (翻转 Phase)，我们用不到，因为没有 warp 特化，或者说
// warp 0是生产者，所有 warp 都是消费者，直接使用 __syncthreads() 同步
__device__ __forceinline__ void mbarrier_arrive(mbarrier_t *mbar) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" ::"r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar))));
}

// 异步等待数据就绪 (自带休眠，不占 ALU 算力)
__device__ __forceinline__ void mbarrier_wait(mbarrier_t *mbar, uint32_t phase) {
    uint32_t is_ready;
    do {
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n"
                     "selp.b32 %0, 1, 0, p;\n"
                     "}\n"
                     : "=r"(is_ready)
                     : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbar))), "r"(phase)
                     : "memory");
    } while (!is_ready);
}

// global to shared::cta 2d TMA 搬运
__device__ __forceinline__ void cp_async_bulk_tensor_2d(
    mbarrier_t *mbar, const void *tmap, const void *smem_ptr, int32_t fast_coord, int32_t slow_coord) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));

    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3}], [%4];\n" ::"r"(smem_addr),
                 "l"(tmap),
                 "r"(fast_coord),
                 "r"(slow_coord),
                 "r"(mbar_addr)
                 : "memory");
}

const int WARP_SIZE = 32;

template <const int BK, typename T>
__device__ __forceinline__ void
ldmatrix_A(uint32_t reg_a[4][4], T (*As)[BK], int warp_id_m, int lane_id, int k_offset) {

    // 4 次 ldmatrix A (4 * 16 = 64 行)
#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
        // ldmatrix x4 读 16x16
        int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
        int a_col = k_offset + (lane_id / 16) * 8;
        uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&As[a_row][SWIZZLE_A(a_row, a_col)]));
        LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
    }
}

template <const int BK, typename T>
__device__ __forceinline__ void
ldmatrix_A_tma(uint32_t reg_a[4][4], T (*As)[BK], int warp_id_m, int lane_id, int k_offset) {

    // 4 次 ldmatrix A (4 * 16 = 64 行)
#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
        int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
        int a_col = k_offset + (lane_id / 16) * 8;
        uint32_t smem_addr =
            static_cast<uint32_t>(__cvta_generic_to_shared(&As[a_row][SWIZZLE_128B_TMA(a_row, a_col)]));
        LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
    }
}

template <const int BN, const int BK, typename T>
__device__ __forceinline__ void
ldmatrix_B(uint32_t reg_b[4][2], T (*Bs)[BN], int warp_id_n, int lane_id, int k_offset) {

    // 4 次 ldmatrix B (4 * 8 = 32 列)
#pragma unroll
    for (int n_idx = 0; n_idx < 4; ++n_idx) {
        // Lane 0~15 的线程读 16 行 (两块 8x8 的首地址)
        int b_row = k_offset + (lane_id % 16);
        int b_col = warp_id_n * 32 + n_idx * 8;

        uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[b_row][SWIZZLE_B(b_row, b_col)]));
        LDMATRIX_X2_TRANS(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
    }
}

template <const int BN, const int BK, typename T>
__device__ __forceinline__ void
ldmatrix_B_tma(uint32_t reg_b[4][2], T (*Bs)[BK][BN / 2], int warp_id_n, int lane_id, int k_offset) {
#pragma unroll
    for (int n_idx = 0; n_idx < 4; ++n_idx) {
        int b_row = k_offset + (lane_id % 16);
        int b_col = warp_id_n * 32 + n_idx * 8;

        // 依靠 chunk_idx 路由，坚决不越界！
        int chunk_idx = b_col / (BN / 2);
        int local_col = b_col % (BN / 2);

        uint32_t smem_addr =
            static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[chunk_idx][b_row][SWIZZLE_128B_TMA(b_row, local_col)]));
        LDMATRIX_X2_TRANS(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
    }
}

template <typename T>
__device__ __forceinline__ void mma_compute(float sum[4][4][4], uint32_t reg_a[4][4], uint32_t reg_b[4][2]) {

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

template <const int BK, typename T>
__device__ __forceinline__ void cp_async_load_A(T (*As)[BK], int load_a_row, int load_a_col, T *global_a_ptr, int k) {

    // cp.async load A，每个线程加载 2 行，跨 64 行
    uint32_t smem_a0 =
        static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row][SWIZZLE_A(load_a_row, load_a_col)]));
    uint32_t smem_a1 =
        static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row + 64][SWIZZLE_A(load_a_row + 64, load_a_col)]));
    // a 矩阵跨 64 行
    CP_ASYNC_CG(smem_a0, global_a_ptr);
    CP_ASYNC_CG(smem_a1, global_a_ptr + 64 * k);
}

template <const int BK, const int BN, typename T>
__device__ __forceinline__ void cp_async_load_B(T (*Bs)[BN], int load_b_row, int load_b_col, T *global_b_ptr, int n) {

    // cp.async load B，每个线程加载 2 行，跨 16 行
    uint32_t smem_b0 =
        static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[load_b_row][SWIZZLE_B(load_b_row, load_b_col)]));
    uint32_t smem_b1 =
        static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[load_b_row + 16][SWIZZLE_B(load_b_row + 16, load_b_col)]));
    // b 矩阵跨 16 行
    CP_ASYNC_CG(smem_b0, global_b_ptr);
    CP_ASYNC_CG(smem_b1, global_b_ptr + 16 * n);
}

template <const int BM, const int BN, typename T>
__device__ __forceinline__ void write_c_via_smem(
    T *c, int by, int bx, int n, float sum[4][4][4], int warp_id_m, int warp_id_n, int lane_id, int tid, T (*Cs)[BN]) {

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
                HALF2(Cs[c_row_0][SWIZZLE_C(c_row_0, c_col)]) = __float22half2_rn(FLOAT2(sum[m_idx][n_idx][0]));
                HALF2(Cs[c_row_2][SWIZZLE_C(c_row_2, c_col)]) = __float22half2_rn(FLOAT2(sum[m_idx][n_idx][2]));
            } else {
                BFLOAT2(Cs[c_row_0][SWIZZLE_C(c_row_0, c_col)]) = __float22bfloat162_rn(FLOAT2(sum[m_idx][n_idx][0]));
                BFLOAT2(Cs[c_row_2][SWIZZLE_C(c_row_2, c_col)]) = __float22bfloat162_rn(FLOAT2(sum[m_idx][n_idx][2]));
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

        FLOAT4(c_block[row * n + col]) = FLOAT4(Cs[row][s_col]);
    }
}

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

    T *global_a_ptr = &a[(by * BM + load_a_row) * k + load_a_col];
    T *global_b_ptr = &b[load_b_row * n + bx * BN + load_b_col];

    // ----------------------------- Prologue 先加载一次As/Bs
    cp_async_load_A<BK>(smem.As[0], load_a_row, load_a_col, global_a_ptr, k);
    cp_async_load_B<BK, BN>(smem.Bs[0], load_b_row, load_b_col, global_b_ptr, n);

    CP_ASYNC_COMMIT_GROUP();
    cp_async_wait_group<0>();
    __syncthreads();

    int read_idx = 0;
    int write_idx = 1;

    // 主循环
    for (int bk = 32; bk < k; bk += BK) {

        // 推进指针
        global_a_ptr += BK;
        global_b_ptr += BK * n;

        // 1. cp.async load A/B
        cp_async_load_A<BK>(smem.As[write_idx], load_a_row, load_a_col, global_a_ptr, k);
        cp_async_load_B<BK, BN>(smem.Bs[write_idx], load_b_row, load_b_col, global_b_ptr, n);

        CP_ASYNC_COMMIT_GROUP();

        // 2. Tensor Core 计算阶段 (k维度分两次，一次 16 个k)
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 16;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // 4 次 ldmatrix A (4 * 16 = 64 行)
            ldmatrix_A<BK>(reg_a, smem.As[read_idx], warp_id_m, lane_id, k_offset);

            // 4 次 ldmatrix B (4 * 8 = 32 列)
            ldmatrix_B<BN, BK>(reg_b, smem.Bs[read_idx], warp_id_n, lane_id, k_offset);

            // MMA 核心运算：4x4 次 m16n8k16
            mma_compute<T>(sum, reg_a, reg_b);
        }

        read_idx ^= 1;
        write_idx ^= 1;

        cp_async_wait_group<0>();
        __syncthreads();
    }
    // ------------------- Epilogue 最后计算一次再写回
#pragma unroll
    for (int k_step = 0; k_step < 2; ++k_step) {
        int k_offset = k_step * 16;

        uint32_t reg_a[4][4];
        uint32_t reg_b[4][2];

        // 4 次 ldmatrix A (4 * 16 = 64 行)
        ldmatrix_A<BK>(reg_a, smem.As[read_idx], warp_id_m, lane_id, k_offset);

        // 4 次 ldmatrix B (4 * 8 = 32 列)
        ldmatrix_B<BN, BK>(reg_b, smem.Bs[read_idx], warp_id_n, lane_id, k_offset);

        // MMA 核心运算：4x4 次 m16n8k16
        mma_compute<T>(sum, reg_a, reg_b);
    }

    write_c_via_smem<BM, BN>(c, by, bx, n, sum, warp_id_m, warp_id_n, lane_id, tid, smem.Cs);
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
        cp_async_load_A<BK>(smem.As[i], load_a_row, load_a_col, global_a_ptr, k);
        cp_async_load_B<BK, BN>(smem.Bs[i], load_b_row, load_b_col, global_b_ptr, n);

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

        // 1. 先发起cp.async load As/Bs 到 load_stage
        cp_async_load_A<BK>(smem.As[load_stage], load_a_row, load_a_col, global_a_ptr, k);
        cp_async_load_B<BK, BN>(smem.Bs[load_stage], load_b_row, load_b_col, global_b_ptr, n);

        CP_ASYNC_COMMIT_GROUP();

        // 2. Tensor Core 计算阶段 (k维度分两次，一次 16 个k)
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 16;
            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // 4 次 ldmatrix A (4 * 16 = 64 行)
            ldmatrix_A<BK>(reg_a, smem.As[compute_stage], warp_id_m, lane_id, k_offset);

            // 4 次 ldmatrix B (4 * 8 = 32 列)
            ldmatrix_B<BN, BK>(reg_b, smem.Bs[compute_stage], warp_id_n, lane_id, k_offset);

            // MMA 核心运算：4x4 次 m16n8k16
            mma_compute<T>(sum, reg_a, reg_b);
        }

        // 推进指针
        global_a_ptr += BK;
        global_b_ptr += BK * n;

        // 流水线往下推一级
        load_stage = (load_stage + 1 == STAGES) ? 0 : load_stage + 1;
        compute_stage = (compute_stage + 1 == STAGES) ? 0 : compute_stage + 1;

        // 保障最早的 group load 好
        cp_async_wait_group<STAGES - 2>();
        __syncthreads();
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
            ldmatrix_A<BK>(reg_a, smem.As[compute_stage], warp_id_m, lane_id, k_offset);

            // 4 次 ldmatrix B (4 * 8 = 32 列)
            ldmatrix_B<BN, BK>(reg_b, smem.Bs[compute_stage], warp_id_n, lane_id, k_offset);

            // MMA 核心运算：4x4 次 m16n8k16
            mma_compute<T>(sum, reg_a, reg_b);
        }

        compute_stage = (compute_stage + 1 == STAGES) ? 0 : compute_stage + 1;
    }

    write_c_via_smem<BM, BN>(c, by, bx, n, sum, warp_id_m, warp_id_n, lane_id, tid, smem.Cs);
}

// -------------------   tma r + mma -------------------
// a block calculate c[128][128]
template <const int BM = 128, const int BN = 128, const int BK = 64, const int STAGES = 3, typename T>
__global__ void hgemm_tma_r_k_stages_kernel(
    __grid_constant__ const CUtensorMap tma_a, __grid_constant__ const CUtensorMap tma_b, T *c, int m, int n, int k) {
    // grid swizzling
    int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8; // 将执行块设置为 8 的宽度

    int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    int by = (linear_id / SWIZZLE_W) % gridDim.y;

    int tid = threadIdx.x; // 0~255
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // 使用动态共享数组
    extern __shared__ __align__(128) uint8_t smem_buf[];
    T(*As)[BM][BK] = reinterpret_cast<T(*)[BM][BK]>(smem_buf);
    T(*Bs)[2][BK][BN / 2] = reinterpret_cast<T(*)[2][BK][BN / 2]>(smem_buf + STAGES * BM * BK * sizeof(T));
    T(*Cs)[BN] = reinterpret_cast<T(*)[BN]>(smem_buf);
    // 把 mbar 放在末尾( 8 字节对齐，3个stages)
    mbarrier_t *mbar = reinterpret_cast<mbarrier_t *>(smem_buf + 98304);

    // 初始化 MBarrier (仅需 tid 0 执行，期待到达次数为 1，因为只有 TMA 会给它发信号)
    if (tid == 0) {
        for (int i = 0; i < STAGES; ++i)
            mbarrier_init(&mbar[i], 1);
    }
    __syncthreads(); // 保证 MBarrier 初始化完毕

    // warp tiling
    // 每个 warp 负责  64 x 32 的 C 矩阵块
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // 寄存器总量：M维4块 * N维4块 * 每块4个寄存器 = 64
    float sum[4][4][4] = {0.f};

    // 每次 TMA 需要搬运的总字节数
    const uint32_t tx_bytes = (BM * BK + BK * BN) * sizeof(T);

    // 只保留一个极其简单的坐标跟踪变量 (因为 TMA 的 Host 描述符里已经知道了跨度)
    int load_k_coord = 0;

    // 1. prologue 加载 STAGES - 1 块
    for (int i = 0; i < STAGES - 1; ++i) {
        if (tid == 0) {
            // 设定这个 mbarrier 需要等多少字节的数据落盘
            mbarrier_expect_tx(&mbar[i], tx_bytes);

            cp_async_bulk_tensor_2d(&mbar[i], &tma_a, As[i], load_k_coord, by * BM);
            cp_async_bulk_tensor_2d(&mbar[i], &tma_b, Bs[i][0], bx * BN, load_k_coord);
            cp_async_bulk_tensor_2d(&mbar[i], &tma_b, Bs[i][1], bx * BN + BN / 2, load_k_coord);
        }
        load_k_coord += BK;
    }

    int load_stage = STAGES - 1;
    int compute_stage = 0;
    int wait_phase = 0; // MBarrier 天然的 0/1 交替相位开关

    // 2. main loop
    for (int bk = (STAGES - 1) * BK; bk < k; bk += BK) {

        // 发起下一轮的 TMA (依然只有 tid 0 干活)
        if (tid == 0) {
            mbarrier_expect_tx(&mbar[load_stage], tx_bytes);
            cp_async_bulk_tensor_2d(&mbar[load_stage], &tma_a, As[load_stage], load_k_coord, by * BM);
            cp_async_bulk_tensor_2d(&mbar[load_stage], &tma_b, Bs[load_stage][0], bx * BN, load_k_coord);
            cp_async_bulk_tensor_2d(&mbar[load_stage], &tma_b, Bs[load_stage][1], bx * BN + BN / 2, load_k_coord);
        }
        load_k_coord += BK;

        // 所有线程：轮询等待当前 compute_stage 的数据被 TMA 搬运完毕
        mbarrier_wait(&mbar[compute_stage], wait_phase);

        // ldmatrix + mma
#pragma unroll
        for (int k_step = 0; k_step < 4; ++k_step) {
            int k_offset = k_step * 16;
            uint32_t reg_a[4][4], reg_b[4][2];

            ldmatrix_A_tma<BK>(reg_a, As[compute_stage], warp_id_m, lane_id, k_offset);
            ldmatrix_B_tma<BN, BK>(reg_b, Bs[compute_stage], warp_id_n, lane_id, k_offset);
            mma_compute<T>(sum, reg_a, reg_b);
        }

        // 直接同步，没有warp 特化，不需要 arrive
        __syncthreads();

        // 状态轮转
        load_stage = (load_stage + 1 == STAGES) ? 0 : load_stage + 1;
        compute_stage = (compute_stage + 1 == STAGES) ? 0 : compute_stage + 1;

        // 完成一次三级流水线，反转 wait_phase
        if (compute_stage == 0)
            wait_phase ^= 1;
    }
    // 3. epilogue 计算 stages-1 次
#pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) {
        // 继续等 TMA
        mbarrier_wait(&mbar[compute_stage], wait_phase);

#pragma unroll
        for (int k_step = 0; k_step < 4; ++k_step) {
            int k_offset = k_step * 16;
            uint32_t reg_a[4][4], reg_b[4][2];

            ldmatrix_A_tma<BK>(reg_a, As[compute_stage], warp_id_m, lane_id, k_offset);
            ldmatrix_B_tma<BN, BK>(reg_b, Bs[compute_stage], warp_id_n, lane_id, k_offset);
            mma_compute<T>(sum, reg_a, reg_b);
        }

        compute_stage = (compute_stage + 1 == STAGES) ? 0 : compute_stage + 1;
        if (compute_stage == 0)
            wait_phase ^= 1;
    }

    // 4. 写回
    write_c_via_smem<BM, BN>(c, by, bx, n, sum, warp_id_m, warp_id_n, lane_id, tid, Cs);
}

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

template <typename T>
inline CUtensorMap
create_tensor_map(T *global_address, uint64_t fast_dim, uint64_t slow_dim, uint32_t fast_box, uint32_t slow_box) {
    CUtensorMap tmap;
    CUtensorMapDataType type =
        std::is_same_v<T, __half> ? CU_TENSOR_MAP_DATA_TYPE_FLOAT16 : CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;

    // TMA 的核心逻辑：第 0 维永远是内存里最连续的维度 (Fastest Changing Dimension)
    uint64_t globalDim[2] = {fast_dim, slow_dim};
    uint64_t globalStrides[1] = {fast_dim * sizeof(T)}; // 外层维度的跨度（字节）
    uint32_t boxDim[2] = {fast_box, slow_box};
    uint32_t elementStrides[2] = {1, 1};

    CUresult res = cuTensorMapEncodeTiled(&tmap,
                                          type,
                                          2, // Tensor Rank (二维矩阵)
                                          global_address,
                                          globalDim,
                                          globalStrides,
                                          boxDim,
                                          elementStrides,
                                          CU_TENSOR_MAP_INTERLEAVE_NONE,
                                          CU_TENSOR_MAP_SWIZZLE_128B, // 对应swizzle
                                          CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    TORCH_CHECK(res == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed!");
    return tmap;
}

// ---------------- tma func binding
#define binding_tiled_tma_func_gen(name)                                                                               \
    void name(torch::Tensor a, torch::Tensor b, torch::Tensor c) {                                                     \
        CHECK_T(a);                                                                                                    \
        CHECK_T(b);                                                                                                    \
        CHECK_T(c);                                                                                                    \
        const int M = a.size(0);                                                                                       \
        const int K = a.size(1);                                                                                       \
        const int N = b.size(1);                                                                                       \
        const int BM = 128;                                                                                            \
        const int BN = 128;                                                                                            \
        const int BK = 64;                                                                                             \
        const int threads_per_block = 256;                                                                             \
        const dim3 blocks_per_grid((N + BN - 1) / BN, (M + BM - 1) / BM);                                              \
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();                                                        \
                                                                                                                       \
        if (a.dtype() == torch::kHalf) {                                                                               \
            CUtensorMap tma_a = create_tensor_map<__half>(reinterpret_cast<__half *>(a.data_ptr()), K, M, BK, BM);     \
            CUtensorMap tma_b = create_tensor_map<__half>(reinterpret_cast<__half *>(b.data_ptr()), N, K, BN / 2, BK); \
                                                                                                                       \
            cudaFuncSetAttribute(                                                                                      \
                name##_kernel<128, 128, 64, 3, __half>, cudaFuncAttributeMaxDynamicSharedMemorySize, 98500);           \
            name##_kernel<128, 128, 64, 3><<<blocks_per_grid, threads_per_block, 98500, stream>>>(                     \
                tma_a, tma_b, reinterpret_cast<__half *>(c.data_ptr()), M, N, K);                                      \
        } else {                                                                                                       \
            CUtensorMap tma_a =                                                                                        \
                create_tensor_map<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16 *>(a.data_ptr()), K, M, BK, BM);       \
            CUtensorMap tma_b =                                                                                        \
                create_tensor_map<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16 *>(b.data_ptr()), N, K, BN / 2, BK);   \
            cudaFuncSetAttribute(                                                                                      \
                name##_kernel<128, 128, 64, 3, __nv_bfloat16>, cudaFuncAttributeMaxDynamicSharedMemorySize, 98500);    \
            name##_kernel<128, 128, 64, 3><<<blocks_per_grid, threads_per_block, 98500, stream>>>(                     \
                tma_a, tma_b, reinterpret_cast<__nv_bfloat16 *>(c.data_ptr()), M, N, K);                               \
        }                                                                                                              \
    }

binding_tiled_func_gen(hgemm_bcf_dbf_rw);
binding_tiled_func_gen(hgemm_k_stages);

binding_tiled_tma_func_gen(hgemm_tma_r_k_stages);

extern void hgemm_cublas(torch::Tensor a, torch::Tensor b, torch::Tensor c);

// binding
#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(hgemm_cublas);
    torch_pybinding_func(hgemm_bcf_dbf_rw);
    torch_pybinding_func(hgemm_k_stages);
    torch_pybinding_func(hgemm_tma_r_k_stages);
}
