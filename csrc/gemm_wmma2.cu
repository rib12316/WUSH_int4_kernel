// ============================================================================
// S-w4: W4A4 GEMM v2 —— v1 的调优版 (逐组 scale 保留, 吞吐提升)
// ============================================================================
// [架构发现实录] 逐组 scale 沿 K 维参与累加: Y=Σ_g sx[m,g]·sw[n,g]·(Σ_k∈g qx·qw)
//   → 数学上不可能放进 epilogue(它只见终态)。CUTLASS s4 快路只支持张量级
//   scale; WUSH 的 FP4 GEMM 是把 scale 喂进主循环(sm_100 blockscaled MMA)。
//   → 精度路径(逐组 scale)只能走"主循环内反量化"的 kernel, 于是优化它。
//
// v1 → v2 的三个改动 (其余结构不变, 降低回归风险):
//   1. TK: 32 → 64 (一个阶段吃 2 个量化组): 同步次数减半, 每阶段计算密度翻倍
//   2. 装载向量化: 每组 16 字节 = 1×uint4 一口读 (v1 逐字节), 指令数 /16
//   3. 反量化写 smem 用 8-half 向量写 (1×uint4)
#include <torch/extension.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

#define TM2 64
#define TN2 64
#define TK2 64          // 2 个量化组/阶段

__global__ void w4a4_gemm_v2_kernel(const uint8_t* __restrict__ qx,   // (M, K/2)
                                    const __nv_bfloat16* __restrict__ sx,  // (M, K/32)
                                    const uint8_t* __restrict__ qw,   // (N, K/2)
                                    const __nv_bfloat16* __restrict__ sw,  // (N, K/32)
                                    __half* __restrict__ y,           // (Mp, Np)
                                    int M, int N, int K)
{
    const int block_m = blockIdx.y * TM2;
    const int block_n = blockIdx.x * TN2;
    const int tid  = threadIdx.x;                  // 256 线程 = 8 warp
    const int warp = tid >> 5;
    // ★★ 根因实录(v2 两轮越界的总答案): 位移量算术。
    //   warp<<4 = warp*16: 8 个 warp 给 {0,16,32,48,64,80,96,112} —— 行 64+ 全越界!
    //   (S-w3 融合kernel用它没错: 那里 block=128 只有 4 个 warp; 这里 block=256 有 8 个)
    //   正确: (warp>>1)<<4 ∈ {0,16,32,48} ×4 行组 × (warp&1)<<5 ∈ {0,32} ×2 列组
    //   = 8 warp 覆盖 64×64, 每 warp 一条 16行×32col 带。
    const int warp_m = (warp >> 1) << 4;           // {0,16,32,48}
    const int warp_n = (warp & 1) << 5;            // {0,32}

    // ★ __align__(16) 实录: __shared__ __half 数组只保证 2B 对齐; 编译器若把它
    //   放在非 16 倍数的 smem 地址, 下面的 uint4 向量读写 = misaligned access
    //   (同样报 illegal memory access, bounds 检查看不出来)。v1 无向量 smem
    //   访问所以从不踩; v2 一引入就必须显式对齐。
    __shared__ __align__(16) __half Xs[TM2][TK2];  // 8KB
    __shared__ __align__(16) __half Ws[TN2][TK2];  // 8KB
    __shared__ float  Cs[TM2][TN2];                // 16KB (仅结尾用)

    // 每 warp: A = 自己的 16 行(逐 k 步重载), B = 2 个 16 列碎片, C = 2 个累加器
    wmma::fragment<wmma::matrix_a,   16, 16, 16, __half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b,   16, 16, 16, __half, wmma::col_major> bf[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf[2];
    #pragma unroll
    for (int j = 0; j < 2; ++j)
        wmma::fill_fragment(cf[j], 0.f);

    const int Kh = K >> 1;
    const int n_groups = K >> 5;

    // 每 CTA 阶段处理 TK2/32 = 2 组; 每组每行 16B → X 装载量 = TM2 行 × 2 组
    for (int g0 = 0; g0 < K; g0 += TK2) {
        // ---- X tile: 一线程一组(uint4 一次读 32 nibble), 展开写 32 half ----
        for (int idx = tid; idx < TM2 * (TK2 / 32); idx += 256) {
            int r = idx / (TK2 / 32), gg = idx % (TK2 / 32);
            int gm = block_m + r;
            if (gm < M) {
                uint4 packed = *reinterpret_cast<const uint4*>(
                    qx + (size_t)gm * Kh + ((g0 >> 1) + gg * 16));
                float s = __bfloat162float(sx[(size_t)gm * n_groups + (g0 >> 5) + gg]);
                __half* dst = &Xs[r][gg * 32];
                const uint8_t* pb = reinterpret_cast<const uint8_t*>(&packed);
                #pragma unroll
                for (int e = 0; e < 32; ++e) {
                    int nib = (e & 1) ? (pb[e >> 1] >> 4) : (pb[e >> 1] & 0xF);
                    int q = (nib << 28) >> 28;
                    dst[e] = __float2half((float)q * s);
                }
            } else {
                #pragma unroll
                for (int c = 0; c < 32; c += 8)
                    *reinterpret_cast<uint4*>(&Xs[r][gg * 32 + c]) = make_uint4(0,0,0,0);
            }
        }
        // ---- W tile: 同款 ----
        for (int idx = tid; idx < TN2 * (TK2 / 32); idx += 256) {
            int r = idx / (TK2 / 32), gg = idx % (TK2 / 32);
            int gn = block_n + r;
            if (gn < N) {
                uint4 packed = *reinterpret_cast<const uint4*>(
                    qw + (size_t)gn * Kh + ((g0 >> 1) + gg * 16));
                float s = __bfloat162float(sw[(size_t)gn * n_groups + (g0 >> 5) + gg]);
                __half* dst = &Ws[r][gg * 32];
                const uint8_t* pb = reinterpret_cast<const uint8_t*>(&packed);
                #pragma unroll
                for (int e = 0; e < 32; ++e) {
                    int nib = (e & 1) ? (pb[e >> 1] >> 4) : (pb[e >> 1] & 0xF);
                    int q = (nib << 28) >> 28;
                    dst[e] = __float2half((float)q * s);
                }
            } else {
                #pragma unroll
                for (int c = 0; c < 32; c += 8)
                    *reinterpret_cast<uint4*>(&Ws[r][gg * 32 + c]) = make_uint4(0,0,0,0);
            }
        }
        __syncthreads();

        // ---- wmma 消费: TK2/16 = 4 个 k 步; 每 warp 16 行 × 2 列碎片 ----
        #pragma unroll
        for (int k = 0; k < TK2 / 16; ++k) {
            wmma::load_matrix_sync(af, &Xs[warp_m][k * 16], TK2);
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(bf[j], &Ws[warp_n + j * 16][k * 16], TK2);
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::mma_sync(cf[j], af, bf[j], cf[j]);
        }
        __syncthreads();
    }

    // ---- 写出: Cs 中转 (wmma 限制同前) ----
    #pragma unroll
    for (int j = 0; j < 2; ++j)
        wmma::store_matrix_sync(&Cs[warp_m][warp_n + j * 16],
                                cf[j], TN2, wmma::mem_row_major);
    __syncthreads();
    const int Np_stride = gridDim.x * TN2;
    for (int idx = tid; idx < TM2 * TN2; idx += 256) {
        int r = idx / TN2, c = idx % TN2;
        int gm = block_m + r, gn = block_n + c;
        if (gm < M && gn < N)
            y[(size_t)gm * Np_stride + gn] = __float2half(Cs[r][c]);
    }
}

torch::Tensor w4a4_gemm_v2(torch::Tensor xq, torch::Tensor xs,
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
    TORCH_CHECK(K % 64 == 0, "K must be multiple of 64 (v2 stage=2 groups)");
    TORCH_CHECK(wq.size(1) == K2 && ws.size(-1) == G && xs.size(-1) == G, "K mismatch");

    auto xq2 = xq.view({-1, K2});
    auto xs2 = xs.reshape({-1, G});
    const int M = (int)xq2.size(0);
    const int N = (int)wq.size(0);
    const int Mp = (M + TM2 - 1) / TM2 * TM2;
    const int Np = (N + TN2 - 1) / TN2 * TN2;
    auto y = torch::zeros({Mp, Np}, torch::dtype(torch::kHalf).device(xq.device()));

    const at::cuda::OptionalCUDAGuard guard(xq.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(xq.device().index());

    dim3 block(256);
    dim3 grid(Np / TN2, Mp / TM2);
    w4a4_gemm_v2_kernel<<<grid, block, 0, stream>>>(
        xq2.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(xs2.data_ptr<at::BFloat16>()),
        wq.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(ws.data_ptr<at::BFloat16>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        M, N, (int)K);

    return y.narrow(0, 0, M).narrow(1, 0, N).contiguous();
}
