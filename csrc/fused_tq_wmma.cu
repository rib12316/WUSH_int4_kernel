// ============================================================================
// 融合 "WUSH变换 + INT4量化" v2 —— WMMA + 多组流水 + Cs 中转(已验证路径)
// ============================================================================
// [P6b] 碎片直读(.x[]布局)实测全错(码一致率0.06, identity-T 也错) → 回退到
// store_matrix_sync 的 Cs 路径(旧 v1 已验证正确), 但保留结构收益:
//   GPS=8 组流水 + Xs/Tgs 双缓冲(预取藏进计算, 单同步/组)
//   FTR=64 + 128线程 + Cs[64][32]fp32 → smem 2×4+2×2+8 = 20KB → 3 CTA/SM
// 定标: σ × 0.35337862372398376 + 1e-8, BF16 roundtrip, clamp [-7,7]
#include <torch/extension.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

#define FTR 64          // 每 CTA 的行数
#define GPS 8           // 组流水深度 (16实测 M2048并行度损失回退 → 8)

__global__ void fused_tq_wmma_kernel(const __nv_bfloat16* __restrict__ x,   // (rows, C*32)
                                     const __nv_bfloat16* __restrict__ t,   // (C, 32, 32)
                                     uint8_t* __restrict__ q_out,           // (rows, C*16)
                                     __nv_bfloat16* __restrict__ s_out,     // (rows, C)
                                     int rows, int C)
{
    const int rb = blockIdx.y * FTR;              // 行块起点
    const int g0 = blockIdx.x * GPS;              // 组段起点
    const int tid = threadIdx.x;                  // 128 线程 = 4 warp
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int warp_m = warp << 4;                 // {0,16,32,48}: 4 warp × 16 行

    __shared__ __half Xs[2][FTR][32];             // 双缓冲 2×4KB
    __shared__ __half Tgs[2][32][32];             // 双缓冲 2×2KB
    __shared__ float Cs[FTR][32];                // 8KB fp32 (WMMA 不支持 float碎片→half smem 的 store, 实测编译错)
                                                  // 主源是 Cs 读写的 smem 放大; 精度代价:
                                                  // 变换值舍入到 fp16, 对拍口径放宽为容差)

    auto load_group = [&](int g, int buf) {
        for (int idx = tid; idx < FTR * 4; idx += 128) {      // 64 行 × 4 段
            int r = idx >> 2, seg = idx & 3;
            int gm = rb + r;
            if (gm < rows) {
                uint4 raw = *reinterpret_cast<const uint4*>(
                    x + (size_t)gm * (C * 32) + g * 32 + seg * 8);
                #pragma unroll
                for (int e = 0; e < 8; ++e)
                    Xs[buf][r][seg * 8 + e] =
                        __float2half(__bfloat162float(reinterpret_cast<__nv_bfloat16*>(&raw)[e]));
            } else {
                #pragma unroll
                for (int e = 0; e < 8; ++e) Xs[buf][r][seg * 8 + e] = __float2half(0.f);
            }
        }
        for (int idx = tid; idx < 128; idx += 128) {          // T[g] 转置, uint4
            int base = idx * 8;
            uint4 raw = *reinterpret_cast<const uint4*>(t + (size_t)g * 1024 + base);
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                int i = base + e;
                Tgs[buf][i & 31][i >> 5] =
                    __float2half(__bfloat162float(reinterpret_cast<__nv_bfloat16*>(&raw)[e]));
            }
        }
    };

    load_group(g0, 0);
    __syncthreads();

    for (int g = g0; g < g0 + GPS && g < C; ++g) {
        const int buf = (g - g0) & 1;             // 相对循环序号(奇数 blockIdx 防 错位)
        if (g + 1 < g0 + GPS && g + 1 < C)
            load_group(g + 1, buf ^ 1);           // 预取下一组到另一缓冲

        // ---- WMMA 变换: Cs = Xs(buf) × Tg^T ----
        wmma::fragment<wmma::matrix_a,   16, 16, 16, __half, wmma::row_major> af[2];
        wmma::fragment<wmma::matrix_b,   16, 16, 16, __half, wmma::row_major> bf[2][2];
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf[2];
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::fill_fragment(cf[j], 0.f);
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            wmma::load_matrix_sync(af[k], &Xs[buf][warp_m][k * 16], 32);
            #pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(bf[k][j], &Tgs[buf][k * 16][j * 16], 32);
        }
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            #pragma unroll
            for (int k = 0; k < 2; ++k)
                wmma::mma_sync(cf[j], af[k], bf[k][j], cf[j]);
        #pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(&Cs[warp_m][j * 16], cf[j], 32, wmma::mem_row_major);
        __syncthreads();                          // Cs 就绪; 下一组的量化读它

        // ---- 量化: 每 warp 管一行, lane = 列 ----
        for (int r = warp; r < FTR; r += 4) {
            int gm = rb + r;
            bool active = gm < rows;
            float v = active ? Cs[r][lane] : 0.f;
            float ps1 = v, ps2 = v * v;
            #pragma unroll
            for (int off = 16; off; off >>= 1) {
                ps1 += __shfl_xor_sync(0xFFFFFFFF, ps1, off);
                ps2 += __shfl_xor_sync(0xFFFFFFFF, ps2, off);
            }
            float c_mean = ps1 / 32.f;
            float var    = ps2 / 32.f - c_mean * c_mean;
            float scale  = (var >= 0.f) ? sqrtf(var) * 0.35337862372398376f + 1e-8f : 1.0f;
            int q = 0;
            if (active) {
                __nv_bfloat16 sb = __float2bfloat16(scale);
                scale = __bfloat162float(sb);      // roundtrip
                q = max(-7, min(7, __float2int_rn(__fdiv_rn(v, scale))));
                if (lane == 0) s_out[(size_t)gm * C + g] = sb;
            }
            int partner = __shfl_xor_sync(0xFFFFFFFF, q, 1);
            if (active && (lane & 1) == 0)
                q_out[(size_t)gm * (C * 16) + g * 16 + (lane >> 1)] =
                    (uint8_t)((q & 0xF) | ((partner & 0xF) << 4));
        }
        __syncthreads();       // 量化读完 Cs 且预取写入完成后, 进入下一组
    }
}

std::vector<torch::Tensor> fused_transform_quant_int4_wmma(torch::Tensor x, torch::Tensor t)
{
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && t.is_cuda() && t.is_contiguous(),
                "x/t must be contiguous CUDA tensors");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && t.scalar_type() == torch::kBFloat16,
                "x/t must be bf16");
    const int64_t K = x.size(-1);
    const int64_t C = K / 32;
    TORCH_CHECK(K % 32 == 0 && t.size(0) == C && t.size(1) == 32 && t.size(2) == 32,
                "T must be (K/32, 32, 32)");

    auto xc = x.contiguous();
    const int rows = (int)(xc.numel() / K);
    const int rp  = (rows + FTR - 1) / FTR * FTR;             // 行补齐到 64
    auto q = torch::zeros({(int64_t)rows * C * 16}, torch::dtype(torch::kUInt8).device(x.device()));
    auto s = torch::zeros({(int64_t)rows * C},      torch::dtype(torch::kBFloat16).device(x.device()));

    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.device().index());

    dim3 grid((unsigned)((C + GPS - 1) / GPS), (unsigned)(rp / FTR));
    fused_tq_wmma_kernel<<<grid, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(xc.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()),
        q.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(s.data_ptr<at::BFloat16>()),
        rows, (int)C);

    std::vector<int64_t> qshape = xc.sizes().vec();  qshape.back() = K / 2;
    std::vector<int64_t> sshape = xc.sizes().vec();  sshape.back() = C;
    return {q.view(torch::IntArrayRef(qshape)), s.view(torch::IntArrayRef(sshape))};
}
