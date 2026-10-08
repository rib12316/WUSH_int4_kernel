// ============================================================================
// S4c: CUTLASS 版 INT4 W4A4 —— 对标 fused_quantize_mx.cu 的架构
// ============================================================================
// [架构对标] 与 WUSH fused_quantize_mx.cu 完全同构的三件套:
//   1. Gemm 类型模板     ← 它的 Gemm_<TileShape, WarpShape, ...>(:57-72)
//   2. GemmRunner        ← 它的 GemmRunner::run(:80-112): Arguments → initialize → ()
//   3. host 入口         ← 它的 fusedQuantizeMx*_host(:117+)
// 差异(格式决定): 它 FP4+Blackwell+量化epilogue; 我们 INT4+Turing+标量alpha。
//
// [模板配置出处] 官方单元测试逐字对标:
//   third_party/cutlass/test/unit/gemm/device/
//     gemm_s4t_s4n_s32n_tensor_op_s32_sm75.cu:60-80
//   (A=int4 行主序 × B=int4 列主序, int32 累加, OpClassTensorOp, Sm75,
//    指令 8x8x32 = Turing 原生 INT4 Tensor Core mma.m8n8k32.s4)
//
// [数据布局的免费午餐] CUTLASS 的 int4b_t 子字节打包: 元素 i 放在字节 i/2 的
//   低 nibble(i 偶) / 高 nibble(i 奇) —— 与我们 S2 kernel 的打包布局完全一致,
//   所以 S2 的 uint8 缓冲可以直接 reinterpret 成 int4b_t*, 零转换。
//   W 的 (N,K) 行主序 = B 矩阵 (K,N) 的列主序 —— 也零转换。
//
// [本站的设计妥协] 逐组(32)scale 在标准 epilogue 里表达不了(它的 alpha 是
//   标量) —— 这正是 WUSH 要自写 epilogue 的原因。本站用张量级 scale:
//   alpha = sx × sw (两个标量), 量化退化为"每张量一组"。
//   尺度收紧的代价由测试的 fp64 仲裁如实反映(精度不如 S3 的分组量化,
//   但架构验证与性能数字是真金白银的)。
#include <torch/extension.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/kernel_launch.h"
#include "cutlass/numeric_types.h"

// ---- 1. Gemm 类型模板 (对标 fused_quantize_mx.cu:57-72 的 Gemm_) ----
using GemmInt4 = cutlass::gemm::device::Gemm<
    cutlass::int4b_t,                 // ElementA: X 的 int4 (M, K)
    cutlass::layout::RowMajor,        // LayoutA
    cutlass::int4b_t,                 // ElementB: W^T 的 int4 (K, N)
    cutlass::layout::ColumnMajor,     // LayoutB: W 的 (N,K) 行主序 = B 的 (K,N) 列主序
    cutlass::half_t,                  // ElementD: fp16 输出
    cutlass::layout::RowMajor,        // LayoutD
    int32_t,                          // ElementAccumulator: 整数累加(Tensor Core s4 原生)
    cutlass::arch::OpClassTensorOp,   // 用 Tensor Core
    cutlass::arch::Sm75,              // Turing
    cutlass::gemm::GemmShape<128, 256, 128>,   // CTA tile  (官方测试同款)
    cutlass::gemm::GemmShape<64, 64, 128>,     // warp tile (官方测试同款)
    cutlass::gemm::GemmShape<8, 8, 32>,        // 指令: mma.m8n8k32.s4
    cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t,
        8,                            // D 向量化对齐 = 128bit/16bit
        int32_t,                      // 中间计算类型: 累加值转 fp32 后乘 alpha
        float>,                       // ElementCompute: alpha 是 fp32 (反量化的乘法在这!)
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2                                 // pipeline stages
>;

// ---- 2. GemmRunner (对标 fused_quantize_mx.cu:74-114; P12 改直接 launch) ----
// [P12] 为什么不走 device::Gemm::run(): 它在 smem ≥ 48KB 时"每次 launch"都调
// cudaFuncSetAttribute(MaxDynamicSharedMemorySize) (device/gemm.h:484) —— 本配置
// SharedStorage 恰 49152B 必触发。该状态变更 API 在 CUDA Graph 捕获期非法, 会让
// run() 在捕获中报 kErrorInternal → TORCH_CHECK 抛异常。修复 = 复刻 initialize+run
// 的 host 部分 (gemm.h:406-448 的 Params 构造是纯 host 算术), 把 attribute 收编为
// 进程一次的 static 守卫 (eager 预热先踩掉), 之后捕获期只剩纯 kernel launch。
struct GemmRunner {
  using OutputOpParams = typename GemmInt4::EpilogueOutputOp::Params;

  bool run(torch::Tensor& y,                       // (M, Np8) fp16, Np8 = N 补齐到 8 倍数
           torch::Tensor const& xq,                // (M, K/2) uint8 = int4 行主序
           torch::Tensor const& wq,                // (N, K/2) uint8 = int4 列主序B
           OutputOpParams const& ep,               // epilogue 参数 (host alpha 或 alpha_ptr)
           torch::Device device,
           int Np8)                                // 对齐后的输出行宽 (= y 的实际列数)
  {
    const int M = (int)(xq.numel() * 2 / (xq.size(-1) * 2));      // 行数 = numel/(K/2)
    const int K = (int)(xq.size(-1) * 2);
    const int N = (int)wq.size(0);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(xq));  // [对标 :104]
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index());  // [对标 :105]

    // [对标 device::Gemm::initialize gemm.h:406-411] grid 形状 = 纯 host 运算
    typename GemmInt4::ThreadblockSwizzle sw;
    cutlass::gemm::GemmCoord grid_shape = sw.get_tiled_shape(
        cutlass::gemm::GemmCoord{M, N, K},
        {GemmInt4::ThreadblockShape::kM, GemmInt4::ThreadblockShape::kN,
         GemmInt4::ThreadblockShape::kK}, 1);

    // [对标 initialize gemm.h:436-448] kernel 级 Params 构造 (纯 host, 零 CUDA 调用)。
    // 注意: Arguments 的 TensorRef 是 const 元素型, kernel Params 吃非 const 迭代器
    // TensorRef —— initialize 原样调 .non_const_ref() 转换, 此处逐字复刻。
    typename GemmInt4::Arguments args{
      cutlass::gemm::GemmCoord{M, N, K},
      {reinterpret_cast<cutlass::int4b_t const*>(xq.data_ptr()), K},   // A (lda=K)
      {reinterpret_cast<cutlass::int4b_t const*>(wq.data_ptr()), K},   // B (ldb=K)
      {reinterpret_cast<cutlass::half_t const*>(y.data_ptr()), Np8},   // C (beta=0, 不读)
      {reinterpret_cast<cutlass::half_t*>(y.data_ptr()), Np8},         // D 真输出, ldm=Np8
      ep,
    };
    typename GemmInt4::GemmKernel::Params params(
        args.problem_size, grid_shape,
        args.ref_A.non_const_ref(), args.ref_B.non_const_ref(),
        args.ref_C.non_const_ref(), args.ref_D,
        args.epilogue, static_cast<int *>(nullptr),
        args.gather_A_indices, args.gather_B_indices, args.scatter_D_indices);

    // 48KB+ 动态 smem 的 opt-in: 进程一次 (graph 捕获期不可再调状态变更 API)
    static bool smem_attr_set = false;
    if (!smem_attr_set) {
      cudaError_t r = cudaFuncSetAttribute(
          cutlass::Kernel<typename GemmInt4::GemmKernel>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          int(sizeof(typename GemmInt4::GemmKernel::SharedStorage)));
      TORCH_CHECK(r == cudaSuccess, "cudaFuncSetAttribute: ", cudaGetErrorString(r));
      smem_attr_set = true;
    }

    // [对标 device::Gemm::run gemm.h:495] 纯 launch, 捕获安全
    cutlass::Kernel<typename GemmInt4::GemmKernel>
      <<<sw.get_grid_shape(params.grid_tiled_shape),
         dim3(GemmInt4::GemmKernel::kThreadCount, 1, 1),
         int(sizeof(typename GemmInt4::GemmKernel::SharedStorage)), stream>>>(params);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "cutlass s4 launch failed");
    return true;
  }
};

// ---- 3. host 入口 (对标 fusedQuantizeMx*_host) ----
torch::Tensor w4a4_gemm_cutlass(torch::Tensor xq, torch::Tensor wq, double alpha)
{
    TORCH_CHECK(xq.is_cuda() && wq.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(xq.scalar_type() == torch::kUInt8 && wq.scalar_type() == torch::kUInt8,
                "packed tensors must be uint8");
    TORCH_CHECK(xq.is_contiguous() && wq.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(wq.dim() == 2, "w must be 2D (N, K/2)");

    // [P12] K 从 wq 推导 (wq 恒 2D); xq 接受 1D flat 或 2D —— 原实现取 xq.size(-1),
    // 1D flat 时会把总 numel 当 K/2 (M=1/K=2·numel 的错位 GEMM)。
    const int64_t K2 = wq.size(1);
    const int M = (int)(xq.numel() / K2);
    TORCH_CHECK(M * K2 == xq.numel(), "xq numel must equal M*K/2 (K from wq)");
    const int N = (int)wq.size(0);
    auto xq2 = xq.view({M, K2});

    // D 的 ldm 需要对齐到 epilogue 向量化宽度(8 half = 128bit):
    // 分配 N 补到 8 倍数的缓冲, 结束后二维切片 —— 生产内核的标准手法
    // (S4c 实录: N=500 时直接用原始 N 会在 epilogue 向量化存储处
    //  触发 misaligned address)
    const int Np8 = (N + 7) / 8 * 8;
    auto y = torch::zeros({M, Np8}, torch::dtype(torch::kHalf).device(xq.device()));
    GemmRunner runner;
    // host 标量 alpha (原路径语义位级不变: 2 参构造 → alpha_ptr=nullptr)
    GemmRunner::OutputOpParams ep{(float)alpha, 0.0f};
    runner.run(y, xq2, wq, ep, xq.device(), Np8);
    return y.narrow(1, 0, N).contiguous();
}

// [P12] graph 化变体: alpha 从设备内存读 (1 元素 fp32 tensor → epilogue alpha_ptr,
// linear_combination.h:176 在 kernel 运行期解引用)。捕获期 alpha 数值可随 replay 变化
// (amax GPU reduce 每步重写该 tensor) —— 数值流全在 GPU, 无 host 同步。
torch::Tensor w4a4_gemm_cutlass_dev(torch::Tensor xq, torch::Tensor wq, torch::Tensor alpha)
{
    TORCH_CHECK(alpha.is_cuda() && alpha.scalar_type() == torch::kFloat32
                && alpha.is_contiguous() && alpha.numel() == 1,
                "alpha must be 1-elem contiguous fp32 CUDA tensor");
    TORCH_CHECK(xq.is_cuda() && wq.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(xq.scalar_type() == torch::kUInt8 && wq.scalar_type() == torch::kUInt8,
                "packed tensors must be uint8");
    TORCH_CHECK(xq.is_contiguous() && wq.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(wq.dim() == 2, "w must be 2D (N, K/2)");

    const int64_t K2 = wq.size(1);              // K 从 wq 推导 (同上, xq 可 1D/2D)
    const int M = (int)(xq.numel() / K2);
    TORCH_CHECK(M * K2 == xq.numel(), "xq numel must equal M*K/2 (K from wq)");
    const int N = (int)wq.size(0);
    auto xq2 = xq.view({M, K2});

    const int Np8 = (N + 7) / 8 * 8;
    auto y = torch::zeros({M, Np8}, torch::dtype(torch::kHalf).device(xq.device()));
    GemmRunner runner;
    GemmRunner::OutputOpParams ep;             // alpha_ptr=nullptr → 运行期设备端读
    ep.alpha_ptr = alpha.data_ptr<float>();
    runner.run(y, xq2, wq, ep, xq.device(), Np8);
    return y.narrow(1, 0, N).contiguous();
}
