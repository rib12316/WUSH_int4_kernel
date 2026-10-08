// ============================================================================
// FAST PATH: 不计精度的算子级极速路线 — per-tensor scale + 原生 s4 Tensor Core
// ============================================================================
// [P12] 用户决策: 算子层面加速优先, 放弃逐组 σ scale 的精度锚点。
// 结构税解除: 无逐组 scale 的标量流 → CUTLASS 原生 s4 GEMM (S4c 实测 6.05×@4096³)。
// 在线侧只差一个极速量化 kernel: 每线程 32 元素 (4×uint4 向量读, 1×uint4 写),
// per-tensor scale 由 host 传入 (alpha = sx·sw 在 GEMM epilogue 一次乘回)。
//   int32 溢出核算: |q|≤7, K≤4096 → |acc| ≤ 7·7·4096 = 200,704 < 2^31 ✓
//   alpha 溢出警示: sx·sw 需 < 3.4e38; 常规 LLM 动态范围 (~1e2) 下安全。
#include <torch/extension.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

__global__ void quant_int4_ts_kernel(const __nv_bfloat16* __restrict__ x,   // (n, ) bf16
                                     uint8_t* __restrict__ q_out,           // (n/2,) packed
                                     float inv_scale,                        // 7/amax (fp32)
                                     long long n32)                          // 32 元素组数
{
    const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n32) return;
    const __nv_bfloat16* src = x + g * 32;
    uint8_t bytes[16];
    #pragma unroll
    for (int h = 0; h < 4; ++h) {              // 4×uint4 = 32 bf16
        uint4 raw = *reinterpret_cast<const uint4*>(src + h * 8);
        __nv_bfloat16* bv = reinterpret_cast<__nv_bfloat16*>(&raw);  // bv[0..7] = 本段 8 元素
        #pragma unroll
        for (int p = 0; p < 4; ++p) {          // 4 字节/段
            const int p0 = p * 2;              // 段内下标 (P12 修 bug: 原用 h*8+p*2 越出 bv[8])
            int q0 = max(-7, min(7, __float2int_rn(__bfloat162float(bv[p0]) * inv_scale)));
            int q1 = max(-7, min(7, __float2int_rn(__bfloat162float(bv[p0 + 1]) * inv_scale)));
            bytes[h * 4 + p] = (uint8_t)((q0 & 0xF) | ((q1 & 0xF) << 4));
        }
    }
    *reinterpret_cast<uint4*>(q_out + g * 16) = *reinterpret_cast<uint4*>(bytes);
}

// per-tensor absmax 量化: x → packed int4。scale = amax/7 由 host 算好传入。
torch::Tensor quant_int4_tensor_scale(torch::Tensor x, double scale)
{
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kBFloat16,
                "x must be contiguous CUDA bf16");
    TORCH_CHECK(x.numel() % 32 == 0, "numel must be multiple of 32");
    auto xc = x.contiguous();
    const long long n32 = xc.numel() / 32;
    auto q = torch::empty({xc.numel() / 2}, torch::dtype(torch::kUInt8).device(x.device()));

    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.device().index());
    float inv_scale = (float)(1.0 / scale);    // = 7/amax (P12 修 bug: 原误写 7.0/scale,
                                               //  整体偏大 7× → ~66% 元素饱和 ±7)
    quant_int4_ts_kernel<<<(unsigned)((n32 + 255) / 256), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(xc.data_ptr()),
        q.data_ptr<uint8_t>(), inv_scale, n32);
    return q;
}

// ============================================================================
// [P12] graph 化变体: inv_scale 从设备内存读 (1 元素 fp32 tensor)。
// 动机: CUDA Graph 捕获期禁止 host 同步 —— amax→inv 必须全程驻留 GPU,
// kernel 在运行期自取 inv_scale (warp 一致的 broadcast 读, 代价可忽略)。
// kernel 体刻意与上面逐字相同 (P1 纪律: 不模板化, eager 版保持字节不变)。
// ============================================================================
__global__ void quant_int4_ts_dev_kernel(const __nv_bfloat16* __restrict__ x,
                                         uint8_t* __restrict__ q_out,
                                         const float* __restrict__ inv_scale_ptr,
                                         long long n32)
{
    const float inv_scale = *inv_scale_ptr;    // 设备端读 scale (graph 安全)
    const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n32) return;
    const __nv_bfloat16* src = x + g * 32;
    uint8_t bytes[16];
    #pragma unroll
    for (int h = 0; h < 4; ++h) {              // 4×uint4 = 32 bf16
        uint4 raw = *reinterpret_cast<const uint4*>(src + h * 8);
        __nv_bfloat16* bv = reinterpret_cast<__nv_bfloat16*>(&raw);  // bv[0..7] = 本段 8 元素
        #pragma unroll
        for (int p = 0; p < 4; ++p) {          // 4 字节/段
            const int p0 = p * 2;
            int q0 = max(-7, min(7, __float2int_rn(__bfloat162float(bv[p0]) * inv_scale)));
            int q1 = max(-7, min(7, __float2int_rn(__bfloat162float(bv[p0 + 1]) * inv_scale)));
            bytes[h * 4 + p] = (uint8_t)((q0 & 0xF) | ((q1 & 0xF) << 4));
        }
    }
    *reinterpret_cast<uint4*>(q_out + g * 16) = *reinterpret_cast<uint4*>(bytes);
}

torch::Tensor quant_int4_tensor_scale_dev(torch::Tensor x, torch::Tensor inv_scale)
{
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kBFloat16,
                "x must be contiguous CUDA bf16");
    TORCH_CHECK(x.numel() % 32 == 0, "numel must be multiple of 32");
    TORCH_CHECK(inv_scale.is_cuda() && inv_scale.scalar_type() == torch::kFloat32
                && inv_scale.is_contiguous() && inv_scale.numel() == 1,
                "inv_scale must be 1-elem contiguous fp32 CUDA tensor");
    const long long n32 = x.numel() / 32;
    auto q = torch::empty({x.numel() / 2}, torch::dtype(torch::kUInt8).device(x.device()));

    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.device().index());
    quant_int4_ts_dev_kernel<<<(unsigned)((n32 + 255) / 256), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        q.data_ptr<uint8_t>(), inv_scale.data_ptr<float>(), n32);
    return q;
}
