// ============================================================================
// P2b: 小 M (≤8) 专用融合 "WUSH变换 + INT4量化" kernel —— decode 路径配套
// ============================================================================
// v1(WMMA) 的 tile 是 64 行: M=1 时 98% 算力在补零上。本 kernel:
//   - warp 认领一个组 g (跨行复用): T[g] 的第 l 行驻 lane l 寄存器(32×bf16)
//   - 行循环(m=0..M): x 组(64B)进 smem → 每 lane 算 x'_l = Σ_i x_i·T[l][i]
//     (smem 广播读, T 行读合并) → 跨 lane σ 归约 → 各 lane 编码 → shuffle 配对
//   - M 行摊销一次 T 载入; M=1 时即为纯 warp-per-group gemv 形态
#include <torch/extension.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define WARPS_PER_CTA 8
#define SM_ROWS 8           // 每 CTA 的行数 (行维并行)

__global__ void fused_tq_smallm_kernel(const __nv_bfloat16* __restrict__ x,   // (M, C*32)
                                       const __nv_bfloat16* __restrict__ t,   // (C, 32, 32)
                                       uint8_t* __restrict__ q_out,           // (M, C*16)
                                       __nv_bfloat16* __restrict__ s_out,     // (M, C)
                                       int M, int C)
{
    const int g = blockIdx.x * WARPS_PER_CTA + (threadIdx.x >> 5);
    if (g >= C) return;
    const int lane = threadIdx.x & 31;
    // [P7] 行维并行: 每 CTA 只管 8 行 (grid.y = ceil(M/8)) —— M=64 时并行度 ×8,
    // 42µs → 目标 ~10µs (T 行重读 ×8 可忽略, 每组 64B×32lane)
    const int m0 = blockIdx.y * SM_ROWS;
    const int m_end = min(M, m0 + SM_ROWS);

    __shared__ __nv_bfloat16 xs[WARPS_PER_CTA][32];      // 每 warp 一份 x 组

    // T[g] 第 lane 行驻寄存器 (32 bf16 = 16 个 bf16x2, 16B/lane 合并读)
    const __nv_bfloat16* trow = t + (size_t)g * 1024 + (size_t)lane * 32;
    float treg[32];
    // [P11] T 行向量化: 32 标量读 → 4×uint4 (每次 8 bf16)。ncu 同款指令流病
    // (v2/v3.1 已修): 每行组 T 装载在 M<=64 只摊销 8 行, 标量流是 36µs 的主源
    {
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            uint4 raw = *reinterpret_cast<const uint4*>(trow + q * 8);
            __nv_bfloat16* bv = reinterpret_cast<__nv_bfloat16*>(&raw);
            #pragma unroll
            for (int e = 0; e < 8; ++e)
                treg[q * 8 + e] = __bfloat162float(bv[e]);
        }
    }

    for (int m = m0; m < m_end; ++m) {
        const int w = threadIdx.x >> 5;
        // x 组进 smem (lane 串行 32 个, 64B; 各 lane 写不同地址 → 无冲突)
        if (lane < 32)
            xs[w][lane] = x[(size_t)m * (C * 32) + g * 32 + lane];
        __syncwarp();

        // x'_lane = Σ_i x_i · T[lane][i]  (smem 广播读: 每 i 步全 lane 同址)
        float xp = 0.f;
        #pragma unroll
        for (int i = 0; i < 32; ++i)
            xp = __fadd_rn(xp, __fmul_rn(__bfloat162float(xs[w][i]), treg[i]));

        // 跨 lane σ: Σx / Σx² 的 warp 归约
        float ps1 = xp, ps2 = xp * xp;
        #pragma unroll
        for (int off = 16; off; off >>= 1) {
            ps1 += __shfl_xor_sync(0xFFFFFFFF, ps1, off);
            ps2 += __shfl_xor_sync(0xFFFFFFFF, ps2, off);
        }
        float c_mean = ps1 * (1.f / 32.f);
        float var    = ps2 * (1.f / 32.f) - c_mean * c_mean;
        float scale  = 1.0f;
        if (var >= 0.f) scale = sqrtf(var) * 0.35337862372398376f + 1e-8f;
        __nv_bfloat16 sb = __float2bfloat16(scale);
        scale = __bfloat162float(sb);                    // roundtrip

        int q = max(-7, min(7, __float2int_rn(__fdiv_rn(xp, scale))));
        int partner = __shfl_xor_sync(0xFFFFFFFF, q, 1);
        if ((lane & 1) == 0)
            q_out[(size_t)m * (C * 16) + g * 16 + (lane >> 1)] =
                (uint8_t)((q & 0xF) | ((partner & 0xF) << 4));
        if (lane == 0)
            s_out[(size_t)m * C + g] = sb;
        __syncwarp();                                     // 下行复用 xs 前的屏障
    }
}

std::vector<torch::Tensor> fused_transform_quant_int4_smallm(torch::Tensor x, torch::Tensor t)
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
    TORCH_CHECK(rows >= 1, "rows must be >= 1");

    auto q = torch::empty({(int64_t)rows * C * 16}, torch::dtype(torch::kUInt8).device(x.device()));
    auto s = torch::empty({(int64_t)rows * C},      torch::dtype(torch::kBFloat16).device(x.device()));

    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.device().index());

    dim3 grid((unsigned)((C + WARPS_PER_CTA - 1) / WARPS_PER_CTA), (unsigned)((rows + SM_ROWS - 1) / SM_ROWS));
    fused_tq_smallm_kernel<<<grid, 32 * WARPS_PER_CTA, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(xc.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()),
        q.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(s.data_ptr<at::BFloat16>()),
        rows, (int)C);

    std::vector<int64_t> qshape = xc.sizes().vec();  qshape.back() = K / 2;
    std::vector<int64_t> sshape = xc.sizes().vec();  sshape.back() = C;
    return {q.view(torch::IntArrayRef(qshape)), s.view(torch::IntArrayRef(sshape))};
}
