// ============================================================================
// S2: INT4 量化 CUDA kernel (v0: 一线程一组, 正确性优先)
// ============================================================================
// 【对照地图】本文件每一部分的上游出处 (WUSH 仓库):
//
//   本文件段落                    对照的 WUSH 文件与行号
//   ─────────────────────────────────────────────────────────────────
//   kernel 整体结构(一组一线程)    epilogue_quant.h:460-576  op_32 的
//                                 AbsMax 分支 —— "absmax→scale→编码→打包"
//                                 的流水顺序完全对标
//   float v[32] 寄存器行          epilogue_quant.h:503  float mat_c[32]
//   absmax 循环                   epilogue_quant.h:549-556
//   scale 计算与存储              epilogue_quant.h:558-561 (它用位掩码存
//                                 E8M0; 我们按 S1 规格改存 fp16 —— 离开
//                                 MX 规范后自己的格式决策)
//   编码+打包循环                 epilogue_quant.h:563-569 (它 ÷scale ×3 后
//                                 调 fp32_vec_to_e2m1 PTX; INT4 均匀格点
//                                 不需要 ×3 头寸技巧, 编码一行搞定)
//   16 字节向量写出               epilogue_quant.h:573  *((float4*)result_ptr)=...
//   舍入/打包的语义权威           quantize_fp4.py:20-46 (rtn_e2m1: 格点表/
//                                 平局取偶/nibble 打包 —— 第4课验证它与
//                                 硬件 cvt.rn.satfinite.e2m1x2 逐位一致)
//   host 封装(守卫/流)            fused_quantize_mx.cu:104-105
//                                 (OptionalCUDAGuard + getCurrentCUDAStream)
//   host 的 M/N/K 计算            fused_quantize_mx.cu:122-124
//                                 (M = A.numel()/32 的"按组数行"思想)
//
// 【与上游的三类本质差异】(不是抄不到位, 是格式不同带来的必然不同):
//   1. 无 GEMM/SMEM/MMA布局代码: 上游的量化发生在 GEMM epilogue 里, 累加器
//      要先经 shared memory 重排(line 488-501 的 acc2smem), 且由 MMA fragment
//      布局决定"每4线程出1个量化线程"(line 511 的 threadIdx.x%4==0);
//      我们没有变换 GEMM, 直接一线程一组。补齐这一半 = S3 的 GEMM。
//   2. scale 格式: 上游 E8M0 位掩码(1字节, 2的幂); 我们 fp16(2字节, 连续值)
//      —— MX 规范强制 vs 自由设计(见 SPEC.md 第0节)。
//   3. 编码: 上游 FP4 需要 fp32_vec_to_e2m1 的 PTX 硬件指令(RNE+饱和一指令
//      双值); INT4 均匀格点, __float2int_rn + clamp 一行搞定, 无需 PTX。
// ============================================================================
#include <torch/extension.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// kernel 本体: 一个线程负责一个 32 元素组
// [对照] 上游没有"这种形态的 kernel"(它的量化在 epilogue 里), 这一层是
//        我们的简化 —— 对应关系见文件头的对照地图第 1 条
// ---------------------------------------------------------------------------
__global__ void quant_int4_kernel(const __nv_bfloat16* __restrict__ x,   // __restrict__: 承诺指针不重叠,
                                  uint8_t* __restrict__ q_out,           // 让编译器放心做读写合并
                                  __nv_bfloat16* __restrict__ s_out,
                                  long long n_groups)
{
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_groups) return;

    // -- 1. 整组载入寄存器 + 串行累加 Σx / Σx² --
    // [P1 对齐] WUSH 论文 INT4 = "Gaussian MSE clipping, BF16 scales, group 32"。
    // 定标为 σ 基(与 epilogue_quant.h:521-534 Quest 模式同式同序):
    //   c_sum1 += x_i; c_sum2 += x_i*x_i  (i 升序串行 —— 参考实现逐位镜像此序)
    float v[32];
    float c_sum1 = 0.f, c_sum2 = 0.f;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        v[i] = __bfloat162float(x[g * 32 + i]);   // bf16→fp32 精确无损
        c_sum1 = __fadd_rn(c_sum1, v[i]);          // 显式 rn: 阻断 nvcc 的 FMA 融合
        c_sum2 = __fadd_rn(c_sum2, __fmul_rn(v[i], v[i]));  // (eager torch 是先乘后加)
    }

    // -- 2. σ 定标: scale = sqrt(var)×C_CLIP + 1e-8, 存 BF16 --
    // C_CLIP = float32(2.473650/7), INT4格点MSE最优 (高斯 16 级量化 MSE 最优截断;
    // 上游 FP4 版除以格点上限 6(epilogue_quant.h:534), INT4 对称上限 7)。
    // var<0(浮点消减) 回退 1.0, 同上游 :533 的 if(var>=0) 守卫。
    float c_mean = c_sum1 / 32.f;                 // /32 精确(2 的幂)
    float var    = __fsub_rn(c_sum2 / 32.f, __fmul_rn(c_mean, c_mean));
    float scale  = 1.0f;
    if (var >= 0.f) scale = __fadd_rn(__fmul_rn(sqrtf(var), 0.35337862372398376f), 1e-8f);

    s_out[g] = __float2bfloat16(scale);           // 存储: BF16 (论文口径)
    scale = __bfloat162float(s_out[g]);           // roundtrip: 编码用舍入后的值

    // -- 3. 舍入成码 + 打包: 对称 RTN, clamp [-7,7] (超截断极值饱和) --
    // __float2int_rn = round-half-to-even(与 torch.round 同语义);
    // 打包: 偶元素低 nibble —— 与 WUSH FP4 pack 字节序一致(第4课 0xF1 实验)
    uint8_t bytes[16];
    #pragma unroll
    for (int i = 0; i < 32; i += 2) {
        int q0 = __float2int_rn(__fdiv_rn(v[i],     scale));
        int q1 = __float2int_rn(__fdiv_rn(v[i + 1], scale));
        q0 = max(-7, min(7, q0));
        q1 = max(-7, min(7, q1));
        bytes[i / 2] = (uint8_t)((q0 & 0xF) | ((q1 & 0xF) << 4));
    }

    // -- 4. 16 字节一次写出(128-bit 存储) --
    // [对照] epilogue_quant.h:573  *((float4*)result_ptr) = *((float4*)result_reg);
    //        同款 128-bit 存储, 同款 16B/组 布局(32 个 4bit 值)
    *reinterpret_cast<uint4*>(q_out + g * 16) = *reinterpret_cast<uint4*>(bytes);
}

// ---------------------------------------------------------------------------
// host 封装: 校验 → 分配 → 设备守卫 → 拿流 → launch → 按输入形状重塑
// [对照] fused_quantize_mx.cu:104-105 (OptionalCUDAGuard + getCurrentCUDAStream,
//        torch 扩展的标准纪律) 与 :122-124 (M = A.numel()/32 的按组数行思想;
//        我们的 n_groups 同义 —— 但它的 M 还要当 GEMM 的行数用, 我们不用)
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> quant_int4(torch::Tensor x)
{
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be contiguous CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
    TORCH_CHECK(x.size(-1) % 32 == 0, "last dim must be multiple of 32");

    auto xc = x.contiguous();
    const long long n_groups = xc.numel() / 32;          // [对照] fused_quantize_mx.cu:122
    auto q = torch::empty({xc.numel() / 2}, torch::dtype(torch::kUInt8).device(x.device()));
    auto s = torch::empty({n_groups},       torch::dtype(torch::kBFloat16).device(x.device()));

    const at::cuda::OptionalCUDAGuard guard(x.device());   // [对照] fused_quantize_mx.cu:104
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.device().index());  // [:105]

    int threads = 256;                                     // v0 随手选; S4 优化时再讲依据
    long long blocks = (n_groups + threads - 1) / threads;
    quant_int4_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(xc.data_ptr()),
        q.data_ptr<uint8_t>(), reinterpret_cast<__nv_bfloat16*>(s.data_ptr<at::BFloat16>()), n_groups);

    // 输出按输入形状重塑(除最后一维减半/除32): API 与参考实现完全对齐,
    // 否则对拍永远死在形状上(第6课教训)
    std::vector<int64_t> qshape = xc.sizes().vec();  qshape.back() = xc.size(-1) / 2;
    std::vector<int64_t> sshape = xc.sizes().vec();  sshape.back() = xc.size(-1) / 32;
    return {q.view(torch::IntArrayRef(qshape)), s.view(torch::IntArrayRef(sshape))};
}
