// ============================================================================
// 融合 "WUSH变换 + INT4量化" v3 —— 纯寄存器 + shuffle 广播 (零 smem staging)
// ============================================================================
// [P8 / ncu 驱动重构] v2(WMMA) 的 l1tex 71.9% 根因 = smem 流量放大:
//   每 4KB 输入产生 ~28KB smem 流量(X staging + 碎片装载 + Cs 读写)。
//   WMMA 存储(float碎片→half)限制又锁死了 Cs 瘦身。 → 彻底脱离 WMMA:
//
// 结构: warp 认领 (组 g, 行段 R 行)。
//   ① T[g] 第 lane 行驻寄存器 treg[32] (每组一次, R 行摊销)
//   ② 每行: x_i 由 shfl 广播 → x'_lane = Σ_i x_i·treg[i]  (32×shfl+FMA, 零 smem)
//   ③ σ: warp 二叉树归约 Σx'/Σx'² (shfl_xor 5 步)
//   ④ 编码 + shfl_xor(1) 配对打包 (偶 nibble 低)
//   双行交错(ILP×2)掩盖 shuffle 链延迟。
// 预算: ~95 warp-instr/行组 → M=2048×K=4096 共 262k 行组 ≈ 25M warp-instr
//   ≈ 119µs issue + 21MB 流量 ≈ 84µs → 目标 ~150µs (v2: 671µs)
#include <torch/extension.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define V3_WARPS 8        // 每 CTA 的 warp 数
#define V3_ROWS  32       // 每 warp 处理的行数

__global__ void fused_tq_v3_kernel(const __nv_bfloat16* __restrict__ x,   // (rows, C*32)
                                   const __nv_bfloat16* __restrict__ t,   // (C, 32, 32)
                                   uint8_t* __restrict__ q_out,           // (rows, C*16)
                                   __nv_bfloat16* __restrict__ s_out,     // (rows, C)
                                   int rows, int C)
{
    const int g = blockIdx.x * V3_WARPS + (threadIdx.x >> 5);
    if (g >= C) return;
    const int lane = threadIdx.x & 31;
    // ★ warp 的 (组, 行段) 必须解耦: CTA 内 8 个 warp 共享同一行段(各 32 行),
    //   各自认领一个组。(实录: 初版 m0 也用 warpId 推导 → 组号与行段耦合,
    //   warp2/3 分到越界行段 → 组2/3 零处理, identity-T 实验抓获)
    const int m0 = blockIdx.y * V3_ROWS;
    const int m_end = min(rows, m0 + V3_ROWS);

    // ① T[g] 的第 lane 行 → treg[32] (fp32 寄存器; lane 间合并读)
    const __nv_bfloat16* trow = t + (size_t)g * 1024 + (size_t)lane * 32;
    float treg[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i)
        treg[i] = __bfloat162float(trow[i]);

    // ②-④ 行循环, 双行交错
    for (int m = m0; m < m_end; m += 2) {
        const bool a1 = (m + 1) < m_end;
        // 每行: lane 读自己列的 x (warp 合并 64B), 由 shfl 广播给全 warp
        // [P8.1] x 打包: 两行 bf16 拼 32bit 走一次 shfl —— ncu 实测 mio_throttle
        // 7.50 主导(shfl 全走 MIO 管线), 跨 lane 流量减半
        unsigned x0b = (unsigned)*reinterpret_cast<const unsigned short*>(
            x + (size_t)m * (C * 32) + g * 32 + lane);
        unsigned x1b = a1 ? (unsigned)*reinterpret_cast<const unsigned short*>(
            x + (size_t)(m + 1) * (C * 32) + g * 32 + lane) : 0u;
        unsigned packed = x0b | (x1b << 16);      // lo=行m, hi=行m+1
        float acc0 = 0.f, acc1 = 0.f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            unsigned pk = __shfl_sync(0xFFFFFFFF, packed, i);
            __nv_bfloat162 bv = *reinterpret_cast<__nv_bfloat162*>(&pk);
            acc0 = __fmaf_rn(__bfloat162float(bv.x), treg[i], acc0);
            acc1 = __fmaf_rn(__bfloat162float(bv.y), treg[i], acc1);
        }
        // σ 树归约 (双行交错, 链间填延迟)
        float s1a = acc0, s2a = acc0 * acc0, s1b = acc1, s2b = acc1 * acc1;
        #pragma unroll
        for (int off = 16; off; off >>= 1) {
            s1a += __shfl_xor_sync(0xFFFFFFFF, s1a, off);
            s2a += __shfl_xor_sync(0xFFFFFFFF, s2a, off);
            s1b += __shfl_xor_sync(0xFFFFFFFF, s1b, off);
            s2b += __shfl_xor_sync(0xFFFFFFFF, s2b, off);
        }
        float qa = 0, qb = 0;
        __nv_bfloat16 sba, sbb;
        {
            float mu = s1a / 32.f;
            float var = s2a / 32.f - mu * mu;
            float sc = (var >= 0.f) ? sqrtf(var) * 0.35337862372398376f + 1e-8f : 1.0f;
            sba = __float2bfloat16(sc);
            sc = __bfloat162float(sba);                     // roundtrip
            qa = max(-7.f, min(7.f, rintf(__fdiv_rn(acc0, sc))));
            if (lane == 0) s_out[(size_t)m * C + g] = sba;
        }
        if (a1) {
            float mu = s1b / 32.f;
            float var = s2b / 32.f - mu * mu;
            float sc = (var >= 0.f) ? sqrtf(var) * 0.35337862372398376f + 1e-8f : 1.0f;
            sbb = __float2bfloat16(sc);
            sc = __bfloat162float(sbb);
            qb = max(-7.f, min(7.f, rintf(__fdiv_rn(acc1, sc))));
            if (lane == 0) s_out[(size_t)(m + 1) * C + g] = sbb;
        }
        // 配对打包: 高 nibble 来自同一行的奇数 lane (shfl_xor 1), 不是相邻行!
        float pa = __shfl_xor_sync(0xFFFFFFFF, qa, 1);
        float pb = __shfl_xor_sync(0xFFFFFFFF, qb, 1);
        if ((lane & 1) == 0) {
            q_out[(size_t)m * (C * 16) + g * 16 + (lane >> 1)] =
                (uint8_t)(((int)qa & 0xF) | (((int)pa & 0xF) << 4));
            if (a1)
                q_out[(size_t)(m + 1) * (C * 16) + g * 16 + (lane >> 1)] =
                    (uint8_t)(((int)qb & 0xF) | (((int)pb & 0xF) << 4));
        }
    }
}

std::vector<torch::Tensor> fused_transform_quant_int4_v3(torch::Tensor x, torch::Tensor t)
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
    const int gy = (rows + V3_ROWS - 1) / V3_ROWS;   // CTA 行覆盖 = V3_ROWS (8 warp 共享)
    auto q = torch::zeros({(int64_t)rows * C * 16}, torch::dtype(torch::kUInt8).device(x.device()));
    auto s = torch::zeros({(int64_t)rows * C},      torch::dtype(torch::kBFloat16).device(x.device()));

    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.device().index());

    dim3 grid((unsigned)((C + V3_WARPS - 1) / V3_WARPS), (unsigned)gy);
    fused_tq_v3_kernel<<<grid, 32 * V3_WARPS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(xc.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()),
        q.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(s.data_ptr<at::BFloat16>()),
        rows, (int)C);

    std::vector<int64_t> qshape = xc.sizes().vec();  qshape.back() = K / 2;
    std::vector<int64_t> sshape = xc.sizes().vec();  sshape.back() = C;
    return {q.view(torch::IntArrayRef(qshape)), s.view(torch::IntArrayRef(sshape))};
}
