// ============================================================================
// S3: W4A4 GEMM v0 —— 反量化 + fp32 FMA, 正确性优先 (不用 Tensor Core)
// ============================================================================
// [对照地图]
//   WUSH 的对应部件是 gemm.cu(Blackwell FP4 tensor-core GEMM, 我们跑不了);
//   我们 v0 实现的是 fp4mm.py 头注释里"pre-Blackwell 降级路径"的手写 CUDA 版:
//   "kernel 内反量化 → fp16/fp32 计算" (fp4mm.py:36-40 论述的两条路径之二)。
//   tile/warp 组织的思路来自 mx_gemm.h 的 TileShape/WarpShape 分层(它是
//   CUTLASS 模板, 我们手写最朴素的 (M,N) 二维网格)。
//
// 核心设计决策: 组内整数点积, 组末才乘 scale
//   Σ_k (qx_k·sx)(qw_k·sw) = Σ_g [ (sx·sw) · (Σ_{k∈g} qx_k·qw_k) ]
//   - 组内 Σ qx·qw 是纯整数: |qx·qw| ≤ 56, 32 项 |和| ≤ 1792 < 2^24,
//     int/fp32 都精确无舍入 —— INT4 才有的红利 (FP4 值非整数做不到)
//   - 一组只付一次"乘 scale"的账
//   - 这也是 S6 原生 INT4 Tensor Core(mma.s4)能工作的数学基础
//
// 对拍口径(实现者决策): 与 S1 参考做**容差对拍**而非逐位 ——
//   两侧 q 值完全相同, 但 fp32 累加顺序/舍入点不同(参考: 逐元素反量化后
//   cuBLAS 矩阵乘; kernel: 组内整数点积+组间 FMA), 原理上必然有最后几 ulp
//   的差异(第5课 FP4 的"累加顺序"教训同款)。另用 fp64 真值做"双方都对"的
//   第三方仲裁, 防止"错得一样"。
#include <torch/extension.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// 4bit 补码符号扩展: 把 bit3 顶到 int 的符号位(bit31)
// [S1 教训] 首版想当然写 <<4>>4 是无效的 —— 位移量 = 容器位宽-数值位宽 = 32-4
__device__ __forceinline__ int sign_extend4(int n)
{
    return (n << 28) >> 28;
}

// ---------------------------------------------------------------------------
// v0 kernel: 一个线程负责一个输出元素 Y[m,n], 自己沿 K 扫
// (内存冗余大 —— 每个 X/W 字节被重复读 N 次或 M 次, v0 不在乎;
//  S4 用 tile+smem+mma 解决, 这里只追求"循环结构和索引全部正确")
// ---------------------------------------------------------------------------
__global__ void w4a4_gemm_v0_kernel(const uint8_t* __restrict__ qx,   // (M, K/2)
                                    const __nv_bfloat16* __restrict__ sx,  // (M, K/32)
                                    const uint8_t* __restrict__ qw,   // (N, K/2)
                                    const __nv_bfloat16* __restrict__ sw,  // (N, K/32)
                                    __half* __restrict__ y,           // (M, N)
                                    int M, int N, int K)
{
    // 二维网格: x 方向铺 N, y 方向铺 M (blockDim 16x16)
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;

    const int n_groups = K / 32;
    const int Kh = K >> 1;                        // 每行打包字节数

    // 行指针提出循环外: 3 个间接寻址都降一维 (编译器本可做, 写明是自我文档化)
    const uint8_t* xrow = qx + (size_t)m * Kh;
    const uint8_t* wrow = qw + (size_t)n * Kh;
    const __nv_bfloat16*  xs   = sx + (size_t)m * n_groups;
    const __nv_bfloat16*  ws   = sw + (size_t)n * n_groups;

    float acc = 0.f;
    for (int g = 0; g < n_groups; ++g) {
        // ---- 组内整数点积 (精确) ----
        int dot = 0;
        const int k0 = g << 5;                    // g*32
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            int k = k0 + i;
            // nibble 提取: 偶元素低4位, 奇元素高4位 (S2 打包布局的镜像)
            uint8_t bx = xrow[k >> 1];
            uint8_t bw = wrow[k >> 1];
            int qx4 = (k & 1) ? (int)(bx >> 4) : (int)(bx & 0xF);
            int qw4 = (k & 1) ? (int)(bw >> 4) : (int)(bw & 0xF);
            dot += sign_extend4(qx4) * sign_extend4(qw4);
        }
        // ---- 组末一次乘 scale ----
        // 写法顺序 dot*(xs*ws): 中间量 (sx·sw) 的舍入只发生一次;
        // 任何顺序都在容差对拍的安全区内, 选它是因为最利于编译器生成 FMA
        acc += (float)dot * __bfloat162float(xs[g]) * __bfloat162float(ws[g]);
    }
    y[(size_t)m * N + n] = __float2half(acc);
}

// ---------------------------------------------------------------------------
// host: x 支持 (..., K) 任意前导维, 内部摊平成 (M, K); w 固定 (N, K)
// ---------------------------------------------------------------------------
torch::Tensor w4a4_gemm(torch::Tensor xq, torch::Tensor xs,
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

    const int64_t K2 = xq.size(-1);               // K/2
    const int64_t K  = K2 * 2;
    const int64_t G  = K / 32;
    TORCH_CHECK(wq.size(1) == K2 && ws.size(-1) == G && xs.size(-1) == G,
                "K mismatch between x/w tensors");

    auto xq2 = xq.view({-1, K2});                 // (M, K/2) —— 摊平前导维
    auto xs2 = xs.reshape({-1, G});
    const int M = (int)xq2.size(0);
    const int N = (int)wq.size(0);

    auto y = torch::empty({M, N}, torch::dtype(torch::kHalf).device(xq.device()));

    const at::cuda::OptionalCUDAGuard guard(xq.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(xq.device().index());

    dim3 block(16, 16);                           // 256 线程/块; v0 不调优
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    w4a4_gemm_v0_kernel<<<grid, block, 0, stream>>>(
        xq2.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(xs2.data_ptr<at::BFloat16>()),
        wq.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(ws.data_ptr<at::BFloat16>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        M, N, (int)K);

    // 还原 x 的前导维: (..., N)
    std::vector<int64_t> yshape = xq.sizes().vec();  yshape.back() = N;
    return y.view(torch::IntArrayRef(yshape));
}
