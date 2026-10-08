// ============================================================================
// P4: 原生 s4 Tensor Core GEMM (m8n8k32) + k组边界scale累加 —— 唯一主线 GEMM
// ============================================================================
// [参考 CUTLASS] 原子直接用 cutlass::arch::Mma<GemmShape<8,8,32>, int4b_t> 的
// PTX (mma.sm75.h:499); 与 CUTLASS 全量 s4 GEMM 的差别只在主循环:
//   CUTLASS: 累加器贯穿整个 K (int32), scale 只能是标量 alpha
//   本kernel: 每个 k组(=32, 恰好一条 m8n8k32)产出独立 int32 部分和, 立即
//             乘 sx[m,g]·sw[n,g] 累加进 fp32 —— 逐组 scale 与 s4 算力兼得
//   (sm100 上 WUSH 由硬件 blockscaled MMA 做同样的事; Turing 上手动做)
//
// 碎片布局 (PTX ISA m8n8k32 s4, 自检函数 s4_atom_selftest 验证):
//   A(8×32): lane T → 行 T/4, k=(T%4)*8..+7 (8 nibble 打包 1×b32, LSB=低k)
//   B(32×8): lane T → 列 n=T/4, k=(T%4)*8..+7
//   C(8×8) : lane T → 行 T/4, 列 (T%4)*2 + {0,1} (2×int32)
#include <torch/extension.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/arch/mma_sm75.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"

using Atom = cutlass::arch::Mma<
    cutlass::gemm::GemmShape<8, 8, 32>, 32,
    cutlass::int4b_t, cutlass::layout::RowMajor,
    cutlass::int4b_t, cutlass::layout::ColumnMajor,
    int, cutlass::layout::RowMajor,
    cutlass::arch::OpMultiplyAddSaturate>;
using FragA = cutlass::Array<cutlass::int4b_t, 8>;
using FragB = cutlass::Array<cutlass::int4b_t, 8>;
using FragC = cutlass::Array<int, 2>;

__device__ __forceinline__ uint32_t pack8nibbles(const int8_t* v)
{
    uint32_t r = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        r |= (uint32_t)(v[i] & 0xF) << (i * 4);
    return r;
}

// ---------------------------------------------------------------------------
// 原子自检: 单 warp, 按上述布局组装 A/B, 结果与 global 写出由 host 对拍
// ---------------------------------------------------------------------------
__global__ void s4_atom_test_kernel(const int8_t* __restrict__ a,   // (8, 32) int8(-8..7)
                                    const int8_t* __restrict__ b,   // (32, 8) int8
                                    int* __restrict__ c)            // (8, 8) int32
{
    const int lane = threadIdx.x & 31;
    // A: 行 T/4, k = (T%4)*8 + i
    int8_t av[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) av[i] = a[(lane >> 2) * 32 + (lane & 3) * 8 + i];
    // B: 列 T/4, k = (T%4)*8 + i   (b 布局 (k, n): b[k*8 + n])
    int8_t bv[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) bv[i] = b[((lane & 3) * 8 + i) * 8 + (lane >> 2)];

    FragA fa; reinterpret_cast<uint32_t&>(fa) = pack8nibbles(av);
    FragB fb; reinterpret_cast<uint32_t&>(fb) = pack8nibbles(bv);
    FragC fc; fc[0] = 0; fc[1] = 0;
    Atom()(fc, fa, fb, fc);

    // C: 行 T/4, 列 (T%4)*2 + {0,1}
    c[(lane >> 2) * 8 + (lane & 3) * 2 + 0] = fc[0];
    c[(lane >> 2) * 8 + (lane & 3) * 2 + 1] = fc[1];
}

torch::Tensor s4_atom_selftest(torch::Tensor a, torch::Tensor b)
{
    auto c = torch::zeros({8, 8}, torch::dtype(torch::kInt32).device(a.device()));
    const at::cuda::OptionalCUDAGuard guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.device().index());
    s4_atom_test_kernel<<<1, 32, 0, stream>>>(
        a.data_ptr<int8_t>(), b.data_ptr<int8_t>(), c.data_ptr<int>());
    return c;
}

// ---------------------------------------------------------------------------
// 主 GEMM: Y[m,n] = Σ_g sx[m,g]·sw[n,g]·(Σ_k∈g qa·qw)   (s4 TC + 逐组scale)
// 布局: CTA tile 128(M)×128(N); 8 warp = 4(M)×2(N), 每 warp 32×64
//        = 4×8 个 m8n8 块; K 每阶段 2 组(64), smem 双缓冲
// ---------------------------------------------------------------------------
#define S4_TN 128
#define S4_TK 64            // 每阶段 k (2 个量化组)
#define S4_STAGES 2
#define S4_ROWSTRIDE (S4_TK / 2 + 16)   // 48B: 32B 数据 + 16B 对齐填充(消 bank conflict)

// [P13] half2 scale 流 —— 数值路径: stage-flush 混合 (用户决策)。
//   逐组 scale 应用从 fp32 标量流 (I2F+FMUL+FFMA, 192 ops/组/线程) 改为
//   cvt(int→half, 精确: |fc|≤1568<2048) + HMUL2/HFMA2 打包 (fp16 管, 实测空闲),
//   flush 窗口内 half2 累加, 窗口边界转存 fp32 运行累加器 —— 官方 fp32 累加纪律。
//   S4_FLUSH_STAGES=1 (每 stage=2 组 flush) 为正确性基线; 收益需加宽窗口
//   (每 stage flush 的 F2F+FADD 恰好吃光打包收益 → B=2 指令数持平, 见 NCU_FINDINGS P13)。
constexpr int S4_FOLD_SHIFT = 8;     // scol 折 2^-8 (staging 时, 纯指数平移): half 范围守卫,
                                     // 实测最坏 hacc 项 7.4 (4300× 余量); 输出端一次 2^+8 展开
constexpr int S4_FLUSH_STAGES = 1;   // 每 N 个 stage flush 一次 hacc→acc (1=每stage; 扫 {1,4,8,64})

// TM 模板: 128(大M, warp 4M×2N, 每warp 32×64) / 64(小M, warp 2M×4N, 每warp 32×32)
// —— M≤64 时 128 行 tile 浪费一半算力, 64 行 tile 消灭它
template <int TM>
__global__ void w4a4_gemm_s4_kernel(const uint8_t* __restrict__ qa,   // (M, K/2)
                                    const __nv_bfloat16* __restrict__ sx, // (M, K/32)
                                    const uint8_t* __restrict__ qw,   // (N, K/2)
                                    const __nv_bfloat16* __restrict__ sw, // (N, K/32)
                                    __half* __restrict__ y,           // (Mp, Np) 行距 Np; splitk 时 nullptr
                                    float* __restrict__ yp,           // splitk: (S, Mp, Np) fp32 部分和; 非splitk为nullptr
                                    int M, int N, int K, int Np,
                                    int k_len)                        // 本 block 的 k 长度 (splitk=K/S, 普通=K)
{
    constexpr int TN = S4_TN;                       // 列向固定 128 (TN=64 实测回退)
    constexpr int WM_SETS = TM / 32;                // warp 的行组数 (每 warp 32 行)
    constexpr int N_SETS = 8 / WM_SETS;             // warp 的列组数
    constexpr int J_TILES = TN / N_SETS / 8;        // 每 warp 的 j 块数 (N 跨度/8)
    const int bm = blockIdx.y * TM;
    const int bn = blockIdx.x * S4_TN;
    const int tid = threadIdx.x;                 // 256 = 8 warp
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm = (warp / N_SETS) << 5;         // TM=128: {0,32,64,96} / TM=64: {0,32}
    const int wn = (warp % N_SETS) * (J_TILES * 8);  // TM=128: {0,64} / TM=64: {0,32,64,96}

    // smem: 行距 48B(消冲突), A/B 双缓冲 + 本阶段 scale 预热
    // [P13] Asx/Bsw 改 half + 组主序 [stage][g][row]: scol 相邻列对可 4B half2 对齐读
    //   (wn + j*8 + (lane&3)*2 恒偶); 半精度只到 smem 暂存层, 全局张量仍 bf16 (论文口径)
    __shared__ __align__(16) uint8_t As[S4_STAGES][TM][S4_ROWSTRIDE];
    __shared__ __align__(16) uint8_t Bs[S4_STAGES][S4_TN][S4_ROWSTRIDE];
    __shared__ __align__(16) __half Asx[S4_STAGES][2][TM];
    __shared__ __align__(16) __half Bsw[S4_STAGES][2][S4_TN];

    const int Kh = K >> 1;
    const int n_groups_total = K >> 5;
    const int n_stages_k = k_len / S4_TK;          // splitk: 本 slice 的阶段数

    // fp32 运行累加器: 4(i)×J_TILES(j)×2 = 64(TM128) / 32(TM64) per thread
    // [P13] + half2 窗口累加器 hacc (S4_FLUSH_STAGES 个 stage 内活跃, flush 进 acc)
    float acc[4][J_TILES][2];
    __half2 hacc[4][J_TILES];
    const __half2 hzero = __float2half2_rn(0.f);
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < J_TILES; ++j) { acc[i][j][0] = 0.f; acc[i][j][1] = 0.f; hacc[i][j] = hzero; }

    // [ncu实录→三级流水] M=64: barrier 2.37 + long_scoreboard 2.54 / 7.26 周期:
    //   global 装载没有真正领先计算。拆装载为 gload(global→regs, 提前2个迭代
    //   发射) / smem_store(regs→smem, 计算时写下一缓冲) —— Turing 无 cp.async 的
    //   标准替代 (Marlin/CUTLASS 2.x 同款手法)。
    struct PR {                                  // 每 thread 的预取寄存器组
        uint4 a[2], b[2];
        __nv_bfloat16 ax[2], bx[2];
        int ar[2], ah[2], br[2], bh[2];
        bool va[2], vb[2];
        int ai, bi;
    };
    auto gload = [&](int ks, PR& r) {
        const int k0 = blockIdx.z * k_len + ks * S4_TK;      // splitk: z 维定位 slice
        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            int ia = tid + c * 256, ib = tid + c * 256;
            r.va[c] = ia < TM * 2;  r.vb[c] = ib < TN * 2;
            r.ar[c] = ia >> 1; r.ah[c] = ia & 1;      // 每行 2 段(16B)
            r.br[c] = ib >> 1; r.bh[c] = ib & 1;
            int gma = bm + r.ar[c], gnb = bn + r.br[c];
            r.a[c] = (r.va[c] && gma < M) ? *reinterpret_cast<const uint4*>(
                          qa + (size_t)gma * Kh + k0 / 2 + r.ah[c] * 16)
                                          : make_uint4(0,0,0,0);
            r.b[c] = (r.vb[c] && gnb < N) ? *reinterpret_cast<const uint4*>(
                          qw + (size_t)gnb * Kh + k0 / 2 + r.bh[c] * 16)
                                          : make_uint4(0,0,0,0);
        }
        r.ai = tid; r.bi = tid;
        int gma = bm + r.ai, gnb = bn + r.bi;
        #pragma unroll
        for (int g = 0; g < 2; ++g) {
            r.ax[g] = (r.ai < TM && gma < M) ? sx[(size_t)gma * n_groups_total + k0 / 32 + g]
                                             : __float2bfloat16(0.f);
            r.bx[g] = (r.bi < TN && gnb < N) ? sw[(size_t)gnb * n_groups_total + k0 / 32 + g]
                                             : __float2bfloat16(0.f);
        }
    };
    auto smem_store = [&](int ks, const PR& r) {
        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            if (r.va[c])
                *reinterpret_cast<uint4*>(&As[ks & 1][r.ar[c]][0] + r.ah[c] * 16) = r.a[c];
            if (r.vb[c])
                *reinterpret_cast<uint4*>(&Bs[ks & 1][r.br[c]][0] + r.bh[c] * 16) = r.b[c];
        }
        // [P13] bf16→half 暂存; scol 折 2^-S4_FOLD_SHIFT (纯指数平移, half 范围守卫),
        //       Asx 不折 —— 输出端统一一次 2^+S4_FOLD_SHIFT 展开
        if (r.ai < TM) {
            Asx[ks & 1][0][r.ai] = __float2half(__bfloat162float(r.ax[0]));
            Asx[ks & 1][1][r.ai] = __float2half(__bfloat162float(r.ax[1]));
        }
        if (r.bi < TN) {
            const float f = 1.0f / (float)(1 << S4_FOLD_SHIFT);
            Bsw[ks & 1][0][r.bi] = __float2half(__bfloat162float(r.bx[0]) * f);
            Bsw[ks & 1][1][r.bi] = __float2half(__bfloat162float(r.bx[1]) * f);
        }
    };
    // ---- 计算体(两条流水路径共用): 当前阶段 2 个 k 组的 mma+scale ----
    // [P13] scale 应用流: fp32 (I2F+FMUL+FFMA)×64元 → cvt(int→half,精确)+HMUL2/HFMA2 打包。
    //   每组: 64 cvt + 32 HMUL2 + 32 HFMA2 (vs 192 fp32 标量); p2 内联懒建不占寄存器数组。
    //   half2 只在 hacc 窗口内累加, flush 进 fp32 acc (stage-flush 混合, 官方 fp32 纪律)。
    auto compute_stage = [&](int ks) {
        #pragma unroll
        for (int g = 0; g < 2; ++g) {
            // 本线程相关 scale: 行 = wm + i*8 + lane/4 (i=0..3); 列 = wn + j*8 + (lane%4)*2 + {0,1}
            __half2 srow2[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                srow2[i] = __half2half2(Asx[ks & 1][g][wm + i * 8 + (lane >> 2)]);
            __half2 scol2[J_TILES];                       // 相邻列对 (已折 2^-FOLD)
            #pragma unroll
            for (int j = 0; j < J_TILES; ++j)
                scol2[j] = *reinterpret_cast<const __half2*>(
                    &Bsw[ks & 1][g][wn + j * 8 + (lane & 3) * 2]);
            // A 碎片: 4 个行块 × 1 (每块行=lane/4, k字节=(lane%4)*4)
            FragA fa[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                reinterpret_cast<uint32_t&>(fa[i]) = *reinterpret_cast<const uint32_t*>(
                    &As[ks & 1][wm + i * 8 + (lane >> 2)][0] + g * 16 + (lane & 3) * 4);
            // B 碎片: J_TILES 个列块; mma + 立即 scale 累加 (half2 打包)
            #pragma unroll
            for (int j = 0; j < J_TILES; ++j) {
                FragB fb;
                reinterpret_cast<uint32_t&>(fb) = *reinterpret_cast<const uint32_t*>(
                    &Bs[ks & 1][wn + j * 8 + (lane >> 2)][0] + g * 16 + (lane & 3) * 4);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    FragC fc; fc[0] = 0; fc[1] = 0;
                    Atom()(fc, fa[i], fb, fc);            // int32 对, 组内精确 (不变)
                    __half2 fc2 = __halves2half2(__int2half_rn(fc[0]), __int2half_rn(fc[1]));
                    hacc[i][j] = __hfma2(fc2, __hmul2(srow2[i], scol2[j]), hacc[i][j]);
                }
            }
        }
    };
    // [P13] flush: hacc 窗口和 → fp32 acc, hacc 清零 (窗口宽度 = S4_FLUSH_STAGES 个 stage)
    auto flush = [&]() {
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            #pragma unroll
            for (int j = 0; j < J_TILES; ++j) {
                float2 f = __half22float2(hacc[i][j]);
                acc[i][j][0] += f.x;
                acc[i][j][1] += f.y;
                hacc[i][j] = hzero;
            }
    };

    PR r0, r1;
    gload(0, r0);
    smem_store(0, r0);
    if constexpr (TM == 128) {
        gload(1, r1);                                   // 三级: 寄存器乒乓, 大M实测 +35%
        __syncthreads();
        for (int ks = 0; ks < n_stages_k; ++ks) {
            if (ks + 2 < n_stages_k) gload(ks + 2, r0);
            if (ks + 1 < n_stages_k) smem_store(ks + 1, r1);
            compute_stage(ks);
            if constexpr (S4_FLUSH_STAGES == 1) flush();
            else if (((ks + 1) & (S4_FLUSH_STAGES - 1)) == 0 || ks == n_stages_k - 1) flush();
            __syncthreads();
            { PR rt = r0; r0 = r1; r1 = rt; }
        }
    } else {
        // 小 TM: 寄存器预算紧(acc 少, 占用率高是它的优势), 三级流水的 +48 寄存器
        // 会把占用打崩(M=64: 1.01→0.75×) → 直通两级
        __syncthreads();
        for (int ks = 0; ks < n_stages_k; ++ks) {
            if (ks + 1 < n_stages_k) { gload(ks + 1, r0); smem_store(ks + 1, r0); }
            compute_stage(ks);
            if constexpr (S4_FLUSH_STAGES == 1) flush();
            else if (((ks + 1) & (S4_FLUSH_STAGES - 1)) == 0 || ks == n_stages_k - 1) flush();
            __syncthreads();
        }
    }

    // ---- 输出: 每线程 fp32 → fp16 直接写 (碎片坐标直写); splitk 时写 fp32 部分和 ----
    // [P13] scol 的 2^-FOLD 在此一次展开 (纯指数, 精确); acc 住在折叠域 (fp32 范围无忧)
    // [P13b] splitk 原子直写: y=zeros + atomicAdd(__half*) (sm70+) —— 省掉 yp 部分和
    //   缓冲的 alloc/memset/sum/copy 链 (~3 launch + S·Mp·Np·4B 流量); 加法序不定
    //   → 逐位不可再现 (容差闸门覆盖); yp 路径保留为回退开关。
    const float unfold = (float)(1 << S4_FOLD_SHIFT);
    const bool atomic_out = (yp == nullptr && gridDim.z > 1);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = bm + wm + i * 8 + (lane >> 2);
        if (row < M) {
            #pragma unroll
            for (int j = 0; j < J_TILES; ++j) {
                int col = bn + wn + j * 8 + (lane & 3) * 2;
                if (col < N) {
                    if (yp != nullptr) {
                        float* dst = yp + ((size_t)blockIdx.z * (gridDim.y * TM) + row) * Np + col;
                        dst[0] = acc[i][j][0] * unfold;
                        if (col + 1 < N) dst[1] = acc[i][j][1] * unfold;
                    } else if (atomic_out) {
                        atomicAdd(y + (size_t)row * Np + col, __float2half(acc[i][j][0] * unfold));
                        if (col + 1 < N)
                            atomicAdd(y + (size_t)row * Np + col + 1, __float2half(acc[i][j][1] * unfold));
                    } else {
                        y[(size_t)row * Np + col + 0] = __float2half(acc[i][j][0] * unfold);
                        if (col + 1 < N) y[(size_t)row * Np + col + 1] = __float2half(acc[i][j][1] * unfold);
                    }
                }
            }
        }
    }
}

torch::Tensor w4a4_gemm_s4(torch::Tensor xq, torch::Tensor xs,
                           torch::Tensor wq, torch::Tensor ws)
{
    TORCH_CHECK(xq.is_cuda() && wq.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(xq.scalar_type() == torch::kUInt8 && wq.scalar_type() == torch::kUInt8,
                "packed tensors must be uint8");
    TORCH_CHECK(xs.scalar_type() == torch::kBFloat16 && ws.scalar_type() == torch::kBFloat16,
                "scales must be bf16");
    TORCH_CHECK(xq.is_contiguous() && xs.is_contiguous() &&
                wq.is_contiguous() && ws.is_contiguous(), "tensors must be contiguous");

    const int64_t K2 = xq.size(-1);
    const int64_t K = K2 * 2, G = K / 32;
    TORCH_CHECK(K % 64 == 0, "K must be multiple of 64");
    auto xq2 = xq.reshape({-1, K2});
    auto xs2 = xs.reshape({-1, G});
    const int M = (int)xq2.size(0);
    const int N = (int)wq.size(0);
    TORCH_CHECK(wq.size(1) == K2 && ws.size(-1) == G, "K mismatch");

    // 形态分派 (含 split-K): M<=8 → TM32 + K切4片 (grid.z=4, CTA数×4 治并行度
    // 缺失; ncu: 64CTA×全K 延迟暴露), fp32 部分和 + torch 归约;
    // M<=2048 → TM64 (波数量化); 其余 TM128。
    // [P10] split-K 全梯队化: 小 M 的共同病 = 网格 CTA 数不足(≤32)延迟暴露。
    //   M≤8:  TM32 + S=4 (128 CTA);  9..64: TM32/64 + S=2 (CTA×2)
    //   门槛条件保证 k_len 仍是 64 的倍数(每阶段 S4_TK=64)
    const int S = (M <= 32 && K % 256 == 0) ? 4 : ((M > 32 && M <= 64 && K % 128 == 0) ? 2 : 1);
    const int tm = (M <= 32) ? 32 : (M <= 2048 ? 64 : 128);
    const int k_len = (int)(K / S);
    const int Mp = (M + tm - 1) / tm * tm;
    const int Np = (N + S4_TN - 1) / S4_TN * S4_TN;
    auto y = torch::zeros({Mp, Np}, torch::dtype(torch::kHalf).device(xq.device()));

    const at::cuda::OptionalCUDAGuard guard(xq.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(xq.device().index());

    dim3 grid((unsigned)(Np / S4_TN), (unsigned)(Mp / tm), (unsigned)S);
    auto* xp = xq2.data_ptr<uint8_t>();
    auto* sp = reinterpret_cast<const __nv_bfloat16*>(xs2.data_ptr<at::BFloat16>());
    auto* wp = wq.data_ptr<uint8_t>();
    auto* swp = reinterpret_cast<const __nv_bfloat16*>(ws.data_ptr<at::BFloat16>());
    auto* yh = reinterpret_cast<__half*>(y.data_ptr<at::Half>());
    if (S > 1) {
        // [P13b] splitk 原子直写: y=zeros (memset 进图) + kernel 内 atomicAdd ——
        //   省掉 yp 部分和链 (alloc/memset/sum/copy ~3 launch + 流量)。
        //   (回退开关: 改传 yp 即恢复 fp32 部分和路径, kernel 侧已保留)
        if (tm == 32)
            w4a4_gemm_s4_kernel<32><<<grid, 256, 0, stream>>>(xp, sp, wp, swp, yh, nullptr, M, N, (int)K, Np, k_len);
        else
            w4a4_gemm_s4_kernel<64><<<grid, 256, 0, stream>>>(xp, sp, wp, swp, yh, nullptr, M, N, (int)K, Np, k_len);
        return y.narrow(0, 0, M).narrow(1, 0, N).contiguous();
    }
    if (tm == 32)
        w4a4_gemm_s4_kernel<32><<<grid, 256, 0, stream>>>(xp, sp, wp, swp, yh, nullptr, M, N, (int)K, Np, (int)K);
    else if (tm == 64)
        w4a4_gemm_s4_kernel<64><<<grid, 256, 0, stream>>>(xp, sp, wp, swp, yh, nullptr, M, N, (int)K, Np, (int)K);
    else
        w4a4_gemm_s4_kernel<128><<<grid, 256, 0, stream>>>(xp, sp, wp, swp, yh, nullptr, M, N, (int)K, Np, (int)K);
    return y.narrow(0, 0, M).narrow(1, 0, N).contiguous();
}
