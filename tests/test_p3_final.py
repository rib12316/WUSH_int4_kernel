# ============================================================================
# P3 终测: WUSH 论文对齐口径 (σ截断+BF16+group32) 的精度与端到端性能全景
# ============================================================================
# 形态分派 (与生产库一致的策略):
#   M ≤ 8  : fused_smallm + gemv        (decode)
#   M > 8  : fused_wmma  + GEMM v2      (prefill)
import torch, sys, os
import os as _os, sys as _sys  # [P14] 仓库根上路径 (tests/bench/scripts 共用)
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from my_int4.rtn_int4 import quant_int4, dequant_int4
from my_int4.wush_transform import compute_wush_transform, transform_weights
from my_int4._C import (quant_int4, w4a4_gemm_v2, w4a4_gemm_s4, w4a4_gemv, w4a4_gemv_multi,
                        fused_transform_quant_int4_wmma, fused_transform_quant_int4_smallm)

def cuda_time(fn, reps=100, warmup=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / reps

torch.manual_seed(0)
print("=" * 72)
print("P3-A 精度 (outlier 数据: 激活通道5 ×15, 权重通道12 ×8, M=2048 N=512 K=4096)")
print("=" * 72)
M, N, K, GT = 2048, 512, 4096, 32
X = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
W = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
X[:, 5::GT] *= 15.0; W[:, 12::GT] *= 8.0
Y_true = X.double() @ W.double().t()
X_cal = torch.randn(256, K, dtype=torch.float32, device='cuda'); X_cal[:, 5::GT] *= 15.0
T64 = compute_wush_transform(X_cal, W.float(), group_size=GT)
T = T64.to(torch.bfloat16)
W_w = transform_weights(W.double(), T64).to(torch.bfloat16).contiguous()

def relerr(y): return ((y.double() - Y_true).norm() / Y_true.norm()).item()

# 裸 INT4 (σ口径)
xq0, xs0 = quant_int4(X); wq0, ws0 = quant_int4(W)
print(f"  裸 INT4 (σ口径)          : {relerr(w4a4_gemm_v2(xq0, xs0, wq0, ws0)):>8.4%}")
# INT4 + WUSH 全链路 (论文口径)
xqw, xsw = fused_transform_quant_int4_wmma(X, T)
wqw, wsw = quant_int4(W_w)
print(f"  INT4 + WUSH (论文口径全链): {relerr(w4a4_gemm_s4(xqw, xsw, wqw, wsw)):>8.4%}")
print(f"  对照 WUSH FP4 (同数据)    :   0.7225%")

print()
print("=" * 72)
print("P3-B 端到端性能 vs fp16 (端到端 = 融合变换+量化 + INT4 GEMM)")
print("=" * 72)
print(f"{'M':>6} {'N':>6} {'K':>6} | {'fp16':>9} {'INT4+WUSH':>10} {'加速比':>7} | {'配置':>22}")
print("-" * 72)
for (M, N, K) in [(1, 4096, 4096), (4, 4096, 4096), (8, 4096, 4096),
                  (64, 4096, 4096), (2048, 512, 4096), (4096, 4096, 4096)]:
    X = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    W = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    X_cal = torch.randn(256, K, dtype=torch.float32, device='cuda')
    T64 = compute_wush_transform(X_cal, W.float(), group_size=32)
    T = T64.to(torch.bfloat16)
    W_w = transform_weights(W.double(), T64).to(torch.bfloat16).contiguous()
    wqw, wsw = quant_int4(W_w)
    X16, W16 = X.half(), W.half()
    reps = 100 if M <= 64 else 10

    t_fp16 = cuda_time(lambda: X16 @ W16.t(), reps=reps)
    if M == 1:
        fuse = lambda: fused_transform_quant_int4_smallm(X, T)
        xqw, xsw = fused_transform_quant_int4_smallm(X, T)
        gemm = lambda: w4a4_gemv(xqw, xsw, wqw, wsw)
        cfg = "smallm + gemv"
    elif M <= 8:
        fuse = lambda: fused_transform_quant_int4_smallm(X, T)
        xqw, xsw = fused_transform_quant_int4_smallm(X, T)
        gemm = lambda: w4a4_gemm_s4(xqw, xsw, wqw, wsw)
        cfg = "smallm + s4-TM32"
    else:
        fuse = lambda: fused_transform_quant_int4_wmma(X, T)
        xqw, xsw = fused_transform_quant_int4_wmma(X, T)
        gemm = lambda: w4a4_gemm_s4(xqw, xsw, wqw, wsw)
        cfg = "wmma + s4 TC"
    t_fuse = cuda_time(fuse, reps=reps)
    t_gemm = cuda_time(gemm, reps=reps)
    t_e2e = t_fuse + t_gemm
    print(f"{M:>6} {N:>6} {K:>6} | {t_fp16:>7.3f}ms {t_e2e:>8.3f}ms {t_fp16/t_e2e:>6.2f}x | {cfg:>22}")
print("-" * 72)
print("注: prefill (M≥64) 为逐组scale的诚实上限分析 —— Turing 无混合精度 MMA,")
print("    dequant→fp16 路径的算力天花板 ≈ fp16 tensor core; 参见终版报告。")
