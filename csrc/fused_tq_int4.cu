// ============================================================================
// S-w2: 融合 "WUSH 变换 + INT4 量化" CUDA kernel
// ============================================================================
// [对照地图] 本文件是 WUSH fused_quantize_wush_mx.cu 在我们架构里的对应物:
//   它: 变换表达成 GEMM(K=32) + 量化 epilogue, 逐块不同的 T 靠 cutlass_patch
//       注入主循环(整个 cutlass_patch/ 目录为此存在)
//   我们: 一线程一组, 直接按组号 g 索引 T[g] —— "逐块不同的变换矩阵"在手写
//       路线里就是一次下标计算(手写路线罕见地更简单)
//
// 语义(与 wush_transform.transform_activations_reference 同源):
//   每组 32 元素: x'_j = Σ_i x_i · T[g][j][i]   (j=0..31, fp32)
//   然后 S2 的量化流水: absmax → scale=amax/7(fp32算,fp16存) → rint → 打包
//
// 逐位一致的三条纪律:
//   1. matvec 用显式 __fmul_rn + __fadd_rn: torch 参考是"乘、加"两次舍入,
//      默认 FMA 合并(-fmad=true)会变成一次舍入 —— 语义不同, 必须显式钉死
//   2. x′ 全程 fp32 寄存器(比参考链路"先落 bf16"更准, 对拍基准是 fp32 镜像)
//   3. 累加顺序 i 升序, 与参考的 32 步循环一致
#include <torch/extension.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void fused_tq_int4_kernel(const __nv_bfloat16* __restrict__ x,   // (rows, K)
                                     const __nv_bfloat16* __restrict__ t,   // (C, 32, 32)
                                     uint8_t* __restrict__ q_out,           // (rows, K/2)
                                     __nv_bfloat16* __restrict__ s_out,            // (rows, C)
                                     int rows, int C)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= rows * C) return;
    int row = gid / C;
    int g   = gid % C;

    // -- 1. 载入该组的 32 个输入 (bf16→fp32 精确) --
    float xv[32];
    const __nv_bfloat16* xg = x + (size_t)row * (C * 32) + g * 32;
    #pragma unroll
    for (int i = 0; i < 32; ++i)
        xv[i] = __bfloat162float(xg[i]);

    // -- 2. 变换: 32 个输出, 每个 = Σ_i x[i]·T[g][j][i] --
    // T[g] 连续存放于 t + g*1024; 逐位一致靠显式 rn 内建(见文件头纪律1)
    // 顺路跟踪 absmax —— 省一次遍历
    const __nv_bfloat16* tg = t + (size_t)g * 1024;
    float xj[32];
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        float acc = 0.f;
        const __nv_bfloat16* tj = tg + (size_t)j * 32;
        #pragma unroll
        for (int i = 0; i < 32; ++i)
            acc = __fadd_rn(acc, __fmul_rn(xv[i], __bfloat162float(tj[i])));
        xj[j] = acc;
    }

    // -- 3. 量化 (P1 对齐: σ 高斯 MSE 截断 + BF16, 同 S2 口径) --
    // v0 原本顺路跟踪 amax; σ 口径需要 Σx/Σx² —— 在 j 循环里改为串行累加
    float c_sum1 = 0.f, c_sum2 = 0.f;
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        c_sum1 = __fadd_rn(c_sum1, xj[j]);                    // 阻断 FMA 融合(逐位镜像)
        c_sum2 = __fadd_rn(c_sum2, __fmul_rn(xj[j], xj[j]));
    }
    float c_mean = c_sum1 / 32.f;
    float var    = __fsub_rn(c_sum2 / 32.f, __fmul_rn(c_mean, c_mean));
    float scale  = 1.0f;
    if (var >= 0.f) scale = __fadd_rn(__fmul_rn(sqrtf(var), 0.35337862372398376f), 1e-8f);
    s_out[gid] = __float2bfloat16(scale);
    scale = __bfloat162float(s_out[gid]);           // roundtrip

    uint8_t bytes[16];
    #pragma unroll
    for (int j = 0; j < 32; j += 2) {
        int q0 = __float2int_rn(__fdiv_rn(xj[j],     scale));
        int q1 = __float2int_rn(__fdiv_rn(xj[j + 1], scale));
        q0 = max(-7, min(7, q0));
        q1 = max(-7, min(7, q1));
        bytes[j / 2] = (uint8_t)((q0 & 0xF) | ((q1 & 0xF) << 4));
    }
    *reinterpret_cast<uint4*>(q_out + (size_t)gid * 16) = *reinterpret_cast<uint4*>(bytes);
}

std::vector<torch::Tensor> fused_transform_quant_int4(torch::Tensor x, torch::Tensor t)
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
    auto q = torch::empty({xc.numel() / 2}, torch::dtype(torch::kUInt8).device(x.device()));
    auto s = torch::empty({(int64_t)rows * C}, torch::dtype(torch::kBFloat16).device(x.device()));

    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.device().index());

    long long total = (long long)rows * C;
    int threads = 256;
    long long blocks = (total + threads - 1) / threads;
    fused_tq_int4_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(xc.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()),
        q.data_ptr<uint8_t>(), reinterpret_cast<__nv_bfloat16*>(s.data_ptr<at::BFloat16>()),
        rows, (int)C);

    std::vector<int64_t> qshape = xc.sizes().vec();  qshape.back() = K / 2;
    std::vector<int64_t> sshape = xc.sizes().vec();  sshape.back() = C;
    return {q.view(torch::IntArrayRef(qshape)), s.view(torch::IntArrayRef(sshape))};
}
