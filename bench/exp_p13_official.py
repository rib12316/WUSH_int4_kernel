# ============================================================================
# P13 官方口径基准: WUSH bench_mxfp4_sm100.py 同款形态 (真实 LLM 层形状 × M 扫描)
# ============================================================================
# 对齐项: 形状集 (Qwen3-8B / Llama-3.1-70B 层), M 扫描, cuda-graph 计时, TFLOP/s 口径。
# 差异项 (本机约束, 如实标注): 基线用 fp16 cuBLAS 而非 BF16 (2060 无快速 bf16 GEMM,
# fp16 为本机等价); 官方数字 4.8-6.6×@M=1024 来自 Blackwell 硬件 blockscaled FP4 MMA。
import torch, sys, os, statistics, gc
import os as _os, sys as _sys  # [P14] 仓库根上路径 (tests/bench/scripts 共用)
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from my_int4.wush_transform import compute_wush_transform, transform_weights
from my_int4.rtn_int4 import quant_int4
from my_int4._C import (w4a4_gemm_s4, w4a4_gemv, w4a4_gemv_multi,
                        fused_transform_quant_int4_v3, fused_transform_quant_int4_smallm)

def cuda_time(fn, reps, warmup=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / reps
def med5(fn, reps): return statistics.median(cuda_time(fn, reps) for _ in range(5))
def graphed(step):
    for _ in range(3): out = step()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out = step()
    return g

def make_chain(X, T, wqw, wsw, M):
    """P13 终态分派: M=1 smallm+gemv / M<=4 smallm+gemv_multi / M<=32 smallm+s4 / 其他 v3.1+s4"""
    if M == 1:
        return lambda: w4a4_gemv(*fused_transform_quant_int4_smallm(X, T), wqw, wsw)
    if M <= 4:
        return lambda: w4a4_gemv_multi(*fused_transform_quant_int4_smallm(X, T), wqw, wsw)
    if M <= 32:
        return lambda: w4a4_gemm_s4(*fused_transform_quant_int4_smallm(X, T), wqw, wsw)
    return lambda: w4a4_gemm_s4(*fused_transform_quant_int4_v3(X, T), wqw, wsw)

torch.manual_seed(0)
SHAPES = [("Qwen3-8B", 4096, 4096), ("Qwen3-8B", 4096, 24576), ("Qwen3-8B", 12288, 4096),
          ("Llama-70B", 8192, 8192), ("Llama-70B", 8192, 57344)]
M_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048, 4096]

for (model, K, N) in SHAPES:
    print(f"\n== {model}  K={K} N={N}  (fp16 cuBLAS 基线; 官方口径为 BF16@Blackwell) ==")
    print(f"{'M':>5} | {'fp16':>9} {'TF/s':>6} | {'int4 e2e':>9} {'TF/s':>6} | {'加速比':>6}")
    print("-" * 62)
    W = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    Xc = torch.randn(256, K, dtype=torch.float32, device='cuda')
    # 巨形状用 fp32 变换 (W.double() 在 6GB 卡上 OOM); 精度不进本表, 仅计性能
    big = N * K > 4 * 4096 * 4096
    Wt = W.float() if big else W.double()
    T64 = compute_wush_transform(Xc, W.float(), group_size=32)
    W_w = transform_weights(Wt, T64.double() if not big else T64).to(torch.bfloat16).contiguous()
    del Wt; gc.collect(); torch.cuda.empty_cache()
    wqw, wsw = quant_int4(W_w)
    del W_w; gc.collect(); torch.cuda.empty_cache()
    W16 = W.half(); del W; gc.collect(); torch.cuda.empty_cache()
    T = T64.to(torch.bfloat16)
    for M in M_SWEEP:
        X = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        X16 = X.half()
        reps = 100 if M <= 4 else (30 if M <= 256 else 8)
        t_fp16 = med5(lambda: X16 @ W16.t(), reps)
        try:
            g = graphed(make_chain(X, T, wqw, wsw, M))
            t = med5(lambda: g.replay(), reps)
        except torch.cuda.OutOfMemoryError:
            print(f"{M:>5} |  OOM"); del X, X16; gc.collect(); torch.cuda.empty_cache(); continue
        tf_fp16 = 2 * M * N * K / (t_fp16 * 1e-3) / 1e12
        tf_int4 = 2 * M * N * K / (t * 1e-3) / 1e12
        print(f"{M:>5} | {t_fp16:>7.3f}ms {tf_fp16:>6.2f} | {t:>7.3f}ms {tf_int4:>6.2f} | {t_fp16/t:>5.2f}x")
        del X, X16, g; gc.collect(); torch.cuda.empty_cache()
    del W16, T, T64, wqw, wsw; gc.collect(); torch.cuda.empty_cache()
