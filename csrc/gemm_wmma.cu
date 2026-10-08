// ============================================================================
// S4: W4A4 GEMM v1 —— tile + shared memory + WMMA fp16 Tensor Core
// ============================================================================
// [对照地图]
//   架构对标 WUSH gemm.cu 的分层: TileShape(CTA) / WarpShape(warp) /
//   InstructionShape(mma 指令) —— 我们分别是 64×64 / 32×32 / 16×16×16(wmma)。
//   "kernel 内反量化进 shared memory → Tensor Core 消费" 的形态,
//   与 fp4mm.py 的 pre-Blackwell 降级路径同一思想, 与 v0 的差别:
//     v0: 一线程一输出, 每个线程重复读整行 → 显存流量放大 N/M 倍
//     v1: CTA 级 tile, X/W 各读一次进 smem, 8 个 warp 共享, wmma 做乘加
//
// 布局: CTA 算 64×64 输出; 8 warp 按 2(M)×2(N) 排, 每 warp 32×32,
//       即 2×2 个 wmma 16×16×16 碎片。K 以 32 为步长(恰 = 一个量化组)推进。
// 边界: M/N 不齐 64 时, 反量化对越界行写 0(零贡献), 输出写进"行/列补齐到16
//       的 padded 缓冲"由 host 切片 —— wmma 的 store 无法逐元素掩码, 这是
//       标准处理手法(等价于 CUTLASS 的 padding)。
#include <torch/extension.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

#define TM 64    // CTA 的 M tile
#define TN 64    // CTA 的 N tile
#define TK 32    // K 步长 = 一个量化组 (反量化与 scale 索引天然对齐)

__global__ void w4a4_gemm_wmma_kernel(const uint8_t* __restrict__ qx,   // (M, K/2)
                                      const __nv_bfloat16* __restrict__ sx,  // (M, K/32)
                                      const uint8_t* __restrict__ qw,   // (N, K/2)
                                      const __nv_bfloat16* __restrict__ sw,  // (N, K/32)
                                      __half* __restrict__ y,           // (Mp, Np) 已补零
                                      int M, int N, int K)
{
    const int block_m = blockIdx.y * TM;          // 本 CTA 的输出行起点
    const int block_n = blockIdx.x * TN;          // 列起点
    const int tid  = threadIdx.x;                 // 256 线程 = 8 warp
    const int warp = tid >> 5;
    // 8 warp 排成 4(M)×2(N): 每 warp 负责输出 tile 的 16×32 区域
    // (1 个 A 碎片 × 2 个 B 碎片 × 2 个 k 步)。★自查记录: 初版排成"2×2 但
    // 位移写错", warp_m 出现 {0,16,32,48} 与 i*16 相加后 Xs[64] 越界踩进 Ws。
    // 覆盖验证: 4 组×16 行 = 64 ✓, 2 组×32 列 = 64 ✓, 无重叠无遗漏
    const int warp_m = (warp >> 1) << 4;          // {0,16,32,48}
    const int warp_n = (warp & 1) << 5;           // {0,32}

    // 反量化暂存区: 本 CTA 的 X/W tile (一个 K chunk)。TK=32 恰好一组,
    // 所以每行只需要一个 scale —— 组索引与 chunk 索引天然重合
    __shared__ __half Xs[TM][TK];                 // 4KB
    __shared__ __half Ws[TN][TK];                 // 4KB

    wmma::fragment<wmma::matrix_a,   16, 16, 16, __half, wmma::row_major> af[2];   // [k步]
    wmma::fragment<wmma::matrix_b,   16, 16, 16, __half, wmma::col_major> bf[2][2];// [j][k步]
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf[2];                    // [j]
    #pragma unroll
    for (int j = 0; j < 2; ++j)
        wmma::fill_fragment(cf[j], 0.f);

    const int n_groups = K >> 5;
    const int Kh = K >> 1;

    for (int g0 = 0; g0 < K; g0 += TK) {          // g0 = 本 chunk 的元素起点(=组号*32)
        // ---- 阶段1: 反量化 X tile (64×32) → smem, 256 线程每人摊 8 个元素 ----
        // 线性 idx → (r, c) 行主序映射: 相邻线程读相邻字节 = 合并访问
        for (int idx = tid; idx < TM * TK; idx += blockDim.x) {
            int r = idx / TK, c = idx - r * TK;
            int gm = block_m + r;
            float v = 0.f;                         // 越界行写 0: 对点积零贡献
            if (gm < M) {
                uint8_t b = qx[(size_t)gm * Kh + ((g0 + c) >> 1)];
                int n4 = ((g0 + c) & 1) ? (int)(b >> 4) : (int)(b & 0xF);
                n4 = (n4 << 28) >> 28;             // sign extend (S1 教训)
                v = (float)n4 * __bfloat162float(sx[(size_t)gm * n_groups + (g0 >> 5)]);
            }
            Xs[r][c] = __float2half(v);
        }
        // ---- 阶段2: 反量化 W tile (64×32) → smem ----
        for (int idx = tid; idx < TN * TK; idx += blockDim.x) {
            int r = idx / TK, c = idx - r * TK;
            int gn = block_n + r;
            float v = 0.f;
            if (gn < N) {
                uint8_t b = qw[(size_t)gn * Kh + ((g0 + c) >> 1)];
                int n4 = ((g0 + c) & 1) ? (int)(b >> 4) : (int)(b & 0xF);
                n4 = (n4 << 28) >> 28;
                v = (float)n4 * __bfloat162float(sw[(size_t)gn * n_groups + (g0 >> 5)]);
            }
            Ws[r][c] = __float2half(v);
        }
        __syncthreads();                            // smem 就绪(也是下轮重写前的屏障)

        // ---- 阶段3: wmma 消费 smem ----
        // A = X 子块(行主序), B = W 子块(存储为 [n][k], wmma 的 B 语义是 (k,n)
        // 列主序 → col_major 标志 + 首地址 &Ws[n0][k0], 步长 TK: 元素(k,n)落在
        // Ws[n0+n][k0+k], 恰是我们的行主序存储 —— 不需要任何转置/拷贝)
        #pragma unroll
        for (int k = 0; k < 2; ++k) {               // TK=32 → 两个 16 深的 k 步
            wmma::load_matrix_sync(af[k], &Xs[warp_m][k * 16], TK);
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(bf[j][k], &Ws[warp_n + j * 16][k * 16], TK);
        }
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            #pragma unroll
            for (int k = 0; k < 2; ++k)
                wmma::mma_sync(cf[j], af[k], bf[j][k], cf[j]);

        __syncthreads();                            // 所有 warp 用完 smem 才能进入下一 chunk 重写
    }

    // ---- 阶段4: acc 写出。wmma 的 store 不支持"float 碎片直接写 __half 内存",
    // 所以先写 fp32 smem, 再整块转 half 写 global(wmma 无逐元素掩码,
    // 因此 y 是补齐到 tile 的 padded 缓冲, 无越界风险) ----
    __shared__ float Cs[TM][TN];                  // 16KB
    #pragma unroll
    for (int j = 0; j < 2; ++j)
        wmma::store_matrix_sync(&Cs[warp_m][warp_n + j * 16],
                                cf[j], (unsigned)TN, wmma::mem_row_major);
    __syncthreads();
    for (int idx = tid; idx < TM * TN; idx += blockDim.x) {
        int r = idx / TN, c = idx - r * TN;
        y[(size_t)(block_m + r) * ((size_t)gridDim.x * TN) + block_n + c] = __float2half(Cs[r][c]);
    }
}

torch::Tensor w4a4_gemm_v1(torch::Tensor xq, torch::Tensor xs,
                           torch::Tensor wq, torch::Tensor ws)
{
    TORCH_CHECK(xq.is_cuda() && wq.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(xq.scalar_type() == torch::kUInt8 && wq.scalar_type() == torch::kUInt8,
                "packed tensors must be uint8");
    TORCH_CHECK(xs.scalar_type() == torch::kBFloat16 && ws.scalar_type() == torch::kBFloat16,
                "scales must be bf16");
    TORCH_CHECK(xq.is_contiguous() && xs.is_contiguous() &&
                wq.is_contiguous() && ws.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(wq.dim() == 2, "w must be 2D (N, K/2)");

    const int64_t K2 = xq.size(-1);
    const int64_t K  = K2 * 2;
    const int64_t G  = K / 32;
    TORCH_CHECK(wq.size(1) == K2 && ws.size(-1) == G && xs.size(-1) == G, "K mismatch");

    auto xq2 = xq.view({-1, K2});
    auto xs2 = xs.reshape({-1, G});
    const int M = (int)xq2.size(0);
    const int N = (int)wq.size(0);

    // 输出补齐到 tile 边界的 padded 缓冲(wmma store 无掩码的标准处理), 切片返回
    const int Mp = (M + TM - 1) / TM * TM;
    const int Np = (N + TN - 1) / TN * TN;
    auto y = torch::zeros({Mp, Np}, torch::dtype(torch::kHalf).device(xq.device()));

    const at::cuda::OptionalCUDAGuard guard(xq.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(xq.device().index());

    dim3 block(256);                                              // 8 warps
    dim3 grid(Np / TN, Mp / TM);
    w4a4_gemm_wmma_kernel<<<grid, block, 0, stream>>>(
        xq2.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(xs2.data_ptr<at::BFloat16>()),
        wq.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(ws.data_ptr<at::BFloat16>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        M, N, (int)K);

    // 切片返回: padded 缓冲是行距 Np 的行主序, 必须做二维 narrow(行和列各切),
    // 不能取"前 M*N 个扁平元素"再 reshape —— 那会把行边界整个错乱
    // (S4 实录: v1 偏离真值 78 的根因就是这一行)。contiguous() 交给调用方语义干净。
    return y.narrow(0, 0, M).narrow(1, 0, N).contiguous();
}
