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

// ------------------------------------------ ldmatrix + mma  ----------------------------------------------------

// a block calculate c[128][128]
template <const int BM = 128, const int BN = 128, const int BK = 32, typename T>
__global__ void hgemm_naive_kernel(T *a, T *b, T *c, int m, int n, int k) {
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

    // A/B 都行优先
    __shared__ T As[BM][BK];
    __shared__ T Bs[BK][BN];

    // warp tiling
    // 每个 warp 负责  64 x 32 的 C 矩阵块
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // 寄存器总量：M维4块 * N维4块 * 每块4个寄存器 = 64
    float sum[4][4][4] = {0.f};

    // 主循环
    for (int bk = 0; bk < k; bk += BK) {

        // 1. cp.async load A
        uint32_t smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row][load_a_col]));
        uint32_t smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_row + 64][load_a_col]));

        T *global_a0 = &a[(by * BM + load_a_row) * k + bk + load_a_col];
        T *global_a1 = &a[(by * BM + load_a_row + 64) * k + bk + load_a_col];

        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);

        // 2. cp.async load B
        uint32_t smem_b0 = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[load_b_row][load_b_col]));
        uint32_t smem_b1 = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[load_b_row + 16][load_b_col]));

        T *global_b0 = &b[(bk + load_b_row) * n + bx * BN + load_b_col];
        T *global_b1 = &b[(bk + load_b_row + 16) * n + bx * BN + load_b_col];

        CP_ASYNC_CG(smem_b0, global_b0);
        CP_ASYNC_CG(smem_b1, global_b1);

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();

        // 3. Tensor Core 计算阶段 (K 分 2 步，一次 16 个k)
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
                uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&As[a_row][a_col]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

            // 4 次 ldmatrix B (4 * 8 = 32 列)
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                // Lane 0~15 的线程恰好覆盖了 16 行 (两块 8x8 的首地址)
                int b_row = k_offset + (lane_id % 16);
                int b_col = warp_id_n * 32 + n_idx * 8;

                uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[b_row][b_col]));
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
            int idx_0 = (c_base_row + t_row) * n + c_base_col + t_col;
            int idx_2 = (c_base_row + t_row + 8) * n + c_base_col + t_col;
            if constexpr (std::is_same_v<T, __half>) {
                HALF2(c[idx_0]) = __float22half2_rn(FLOAT2(sum[m_idx][n_idx][0]));
                HALF2(c[idx_2]) = __float22half2_rn(FLOAT2(sum[m_idx][n_idx][2]));
            } else {
                BFLOAT2(c[idx_0]) = __float22bfloat162_rn(FLOAT2(sum[m_idx][n_idx][0]));
                BFLOAT2(c[idx_2]) = __float22bfloat162_rn(FLOAT2(sum[m_idx][n_idx][2]));
            }
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

binding_tiled_func_gen(hgemm_naive);

extern void hgemm_cublas(torch::Tensor a, torch::Tensor b, torch::Tensor c);

// binding
#define torch_pybinding_func(f) m.def(#f, &f, #f)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_pybinding_func(hgemm_cublas);
    torch_pybinding_func(hgemm_naive);
}
