// ============================================================================
// P2a: W4A4 decode 专用 gemv kernel (M=1)
// ============================================================================
// 场景: LLM 自回归解码 —— 每步 1 行激活 × 全部权重。瓶颈 = 权重显存带宽,
// INT4 权重流量 = fp16 的 1/4 → 这是 W4A4 在 pre-Blackwell 上确定能赢的阵地。
//
// 设计 (带宽优先):
//   - 每 warp 负责一个输出 y[n]; grid.y 分 N
//   - x (1, K) 与 sx (K/32) 提前进 smem, 全 CTA 共享 (激活流量可忽略)
//   - lane l 认领组 g = l, l+32, ...: 每组一次 16B uint4 读 W 行
//     → warp 32 lane 连续读 512B, 完美合并
//   - 组内 32 nibble 整数点积 (int 域精确, S3 的架构红利), 组末 fp32 乘
//     sx[g]·sw[n,g]; 跨组跨 lane 用 warp shuffle 归约; lane0 写 fp16
#include <torch/extension.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define GEMV_ROWS_PER_CTA 8            // 每 CTA 的 warp 数 = 处理的 n 行数

__global__ void w4a4_gemv_kernel(const uint8_t* __restrict__ qx,     // (K/2)
                                 const __nv_bfloat16* __restrict__ sx, // (K/32)
                                 const uint8_t* __restrict__ qw,     // (N, K/2)
                                 const __nv_bfloat16* __restrict__ sw, // (N, K/32)
                                 __half* __restrict__ y,             // (N)
                                 int N, int K)
{
    const int n = blockIdx.x * GEMV_ROWS_PER_CTA + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int Kh = K >> 1;
    const int n_groups = K >> 5;

    // x 打包字节 + sx 进 smem (K/2 ≤ 动态上限; K=4096 → 2KB + 128B)
    extern __shared__ uint8_t smem[];
    uint8_t* sxq = smem;                       // K/2 字节
    __nv_bfloat16* sxs = reinterpret_cast<__nv_bfloat16*>(smem + Kh);  // K/32 × 2B
    for (int i = threadIdx.x; i < Kh; i += blockDim.x) sxq[i] = qx[i];
    for (int i = threadIdx.x; i < n_groups; i += blockDim.x) sxs[i] = sx[i];
    __syncthreads();
    if (n >= N) return;

    // 每 lane 认领组 g = lane + 32t; 组内整数点积 + 组末 scale
    float acc = 0.f;
    const uint8_t* wrow = qw + (size_t)n * Kh;
    const __nv_bfloat16* swr = sw + (size_t)n * n_groups;
    for (int g = lane; g < n_groups; g += 32) {
        uint4 wb = *reinterpret_cast<const uint4*>(wrow + g * 16);
        uint4 xb = *reinterpret_cast<const uint4*>(sxq + g * 16);
        int dot = 0;
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            uint32_t w32 = (&wb.x)[b], x32 = (&xb.x)[b];
            #pragma unroll
            for (int nb = 0; nb < 8; ++nb) {
                int qw4 = (int)((w32 >> (nb * 4)) & 0xF);
                int qx4 = (int)((x32 >> (nb * 4)) & 0xF);
                qw4 = (qw4 << 28) >> 28;                 // sign extend (S1 教训)
                qx4 = (qx4 << 28) >> 28;
                dot += qw4 * qx4;
            }
        }
        acc += (float)dot * __bfloat162float(sxs[g]) * __bfloat162float(swr[g]);
    }

    // warp 归约 + lane0 写出
    #pragma unroll
    for (int off = 16; off; off >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, off);
    if (lane == 0) y[n] = __float2half(acc);
}

// ---------------------------------------------------------------------------
// 多行版 (M ≤ 8): 每 warp 一个 n, W 行 16B 一次载入供 M 行共享 —— W 流量
// 不随 M 增长(decode batch 的关键: 权重带宽是唯一瓶颈, 激活全部驻 smem)
// ---------------------------------------------------------------------------
template <int M>
__global__ void w4a4_gemv_multi_kernel(const uint8_t* __restrict__ qx,     // (M, K/2)
                                       const __nv_bfloat16* __restrict__ sx, // (M, K/32)
                                       const uint8_t* __restrict__ qw,     // (N, K/2)
                                       const __nv_bfloat16* __restrict__ sw, // (N, K/32)
                                       __half* __restrict__ y,             // (M, N)
                                       int N, int K)
{
    const int n = blockIdx.x * GEMV_ROWS_PER_CTA + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int Kh = K >> 1;
    const int n_groups = K >> 5;

    extern __shared__ uint8_t smem[];
    uint8_t* sxq = smem;                                     // M × K/2 (打包态)
    __nv_bfloat16* sxs = reinterpret_cast<__nv_bfloat16*>(smem + (size_t)M * Kh);
    for (int i = threadIdx.x; i < M * Kh; i += blockDim.x) sxq[i] = qx[i];
    for (int i = threadIdx.x; i < M * n_groups; i += blockDim.x) sxs[i] = sx[i];
    __syncthreads();
    if (n >= N) return;

    float acc[M];                                            // M 模板参数 → 寄存器驻留
    #pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;                // (运行时M+动态索引会 spill 到 local)
    const uint8_t* wrow = qw + (size_t)n * Kh;
    const __nv_bfloat16* swr = sw + (size_t)n * n_groups;
    for (int g = lane; g < n_groups; g += 32) {
        uint4 wb = *reinterpret_cast<const uint4*>(wrow + g * 16);
        int wv[32];
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            uint32_t w32 = (&wb.x)[b];
            #pragma unroll
            for (int nb = 0; nb < 8; ++nb) {
                int qw4 = (int)((w32 >> (nb * 4)) & 0xF);
                wv[b * 8 + nb] = (qw4 << 28) >> 28;          // sign extend
            }
        }
        float swn = __bfloat162float(swr[g]);
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            uint4 xb = *reinterpret_cast<const uint4*>(sxq + (size_t)m * Kh + g * 16);
            int dot = 0;
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                uint32_t x32 = (&xb.x)[b];
                #pragma unroll
                for (int nb = 0; nb < 8; ++nb) {
                    int qx4 = (int)((x32 >> (nb * 4)) & 0xF);
                    qx4 = (qx4 << 28) >> 28;
                    dot += wv[b * 8 + nb] * qx4;
                }
            }
            acc[m] += (float)dot * __bfloat162float(sxs[(size_t)m * n_groups + g]) * swn;
        }
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1)
        #pragma unroll
        for (int m = 0; m < M; ++m)
            acc[m] += __shfl_xor_sync(0xFFFFFFFF, acc[m], off);
    if (lane == 0)
        #pragma unroll
        for (int m = 0; m < M; ++m)
            y[(size_t)m * N + n] = __float2half(acc[m]);
}

torch::Tensor w4a4_gemv_multi(torch::Tensor xq, torch::Tensor xs,
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
    auto xq2 = xq.reshape({-1, K2});
    auto xs2 = xs.reshape({-1, G});
    const int M = (int)xq2.size(0);
    TORCH_CHECK(M >= 1 && M <= 8, "gemv_multi expects 1<=M<=8");
    TORCH_CHECK(wq.size(1) == K2 && ws.size(-1) == G, "K mismatch");

    const int N = (int)wq.size(0);
    auto y = torch::empty({M, N}, torch::dtype(torch::kHalf).device(xq.device()));

    const at::cuda::OptionalCUDAGuard guard(xq.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(xq.device().index());

    const int smem_bytes = (int)(M * K2 + (size_t)M * G * 2);
    dim3 block(32 * GEMV_ROWS_PER_CTA);
    dim3 grid((N + GEMV_ROWS_PER_CTA - 1) / GEMV_ROWS_PER_CTA);
    auto* xp = xq2.data_ptr<uint8_t>();
    auto* sp = reinterpret_cast<const __nv_bfloat16*>(xs2.data_ptr<at::BFloat16>());
    auto* wp = wq.data_ptr<uint8_t>();
    auto* swp = reinterpret_cast<const __nv_bfloat16*>(ws.data_ptr<at::BFloat16>());
    auto* yp = reinterpret_cast<__half*>(y.data_ptr<at::Half>());
    // 模板分派: M 编译期定长 (1..8), 避免 acc[] 动态索引 spill
    switch (M) {
        case 1: w4a4_gemv_multi_kernel<1><<<grid, block, smem_bytes, stream>>>(xp, sp, wp, swp, yp, N, (int)K); break;
        case 2: w4a4_gemv_multi_kernel<2><<<grid, block, smem_bytes, stream>>>(xp, sp, wp, swp, yp, N, (int)K); break;
        case 3: w4a4_gemv_multi_kernel<3><<<grid, block, smem_bytes, stream>>>(xp, sp, wp, swp, yp, N, (int)K); break;
        case 4: w4a4_gemv_multi_kernel<4><<<grid, block, smem_bytes, stream>>>(xp, sp, wp, swp, yp, N, (int)K); break;
        case 5: w4a4_gemv_multi_kernel<5><<<grid, block, smem_bytes, stream>>>(xp, sp, wp, swp, yp, N, (int)K); break;
        case 6: w4a4_gemv_multi_kernel<6><<<grid, block, smem_bytes, stream>>>(xp, sp, wp, swp, yp, N, (int)K); break;
        case 7: w4a4_gemv_multi_kernel<7><<<grid, block, smem_bytes, stream>>>(xp, sp, wp, swp, yp, N, (int)K); break;
        case 8: w4a4_gemv_multi_kernel<8><<<grid, block, smem_bytes, stream>>>(xp, sp, wp, swp, yp, N, (int)K); break;
    }
    return y;
}

torch::Tensor w4a4_gemv(torch::Tensor xq, torch::Tensor xs,
                        torch::Tensor wq, torch::Tensor ws)
{
    TORCH_CHECK(xq.is_cuda() && wq.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(xq.scalar_type() == torch::kUInt8 && wq.scalar_type() == torch::kUInt8,
                "packed tensors must be uint8");
    TORCH_CHECK(xs.scalar_type() == torch::kBFloat16 && ws.scalar_type() == torch::kBFloat16,
                "scales must be bf16");
    TORCH_CHECK(xq.is_contiguous() && xs.is_contiguous() &&
                wq.is_contiguous() && ws.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(xq.dim() <= 2 && xq.numel() == xq.size(-1), "gemv expects a single row");
    const int64_t K2 = xq.size(-1);
    const int64_t K = K2 * 2, G = K / 32;
    TORCH_CHECK(wq.size(1) == K2 && ws.size(-1) == G, "K mismatch");

    const int N = (int)wq.size(0);
    auto y = torch::empty({N}, torch::dtype(torch::kHalf).device(xq.device()));

    const at::cuda::OptionalCUDAGuard guard(xq.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(xq.device().index());

    const int smem_bytes = (int)(K2 + G * 2);
    dim3 block(32 * GEMV_ROWS_PER_CTA);
    dim3 grid((N + GEMV_ROWS_PER_CTA - 1) / GEMV_ROWS_PER_CTA);
    w4a4_gemv_kernel<<<grid, block, smem_bytes, stream>>>(
        xq.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(xs.data_ptr<at::BFloat16>()),
        wq.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(ws.data_ptr<at::BFloat16>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        N, (int)K);
    return y;
}
