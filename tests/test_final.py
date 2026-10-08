# ============================================================================
# S-w4 收官: v2 GEMM 正确性 + 最终性能总表 + 全链路回归
# ============================================================================
import torch, sys, os
import os as _os, sys as _sys  # [P14] 仓库根上路径 (tests/bench/scripts 共用)
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from my_int4.rtn_int4 import quant_int4, dequant_int4, w4a4_matmul_reference
from my_int4.wush_transform import compute_wush_transform, transform_weights
from my_int4._C import (quant_int4, w4a4_gemm, w4a4_gemm_v1, w4a4_gemm_v2,
                        w4a4_gemm_cutlass,
                        fused_transform_quant_int4, fused_transform_quant_int4_wmma)

def cuda_time(fn, reps=30, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / reps

print("== v2 GEMM 正确性 (fp64 仲裁) ==")
torch.manual_seed(0)
for (M, N, K) in [(64, 128, 128), (257, 33, 192), (130, 512, 4096)]:
    X = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    W = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    xq, xs = quant_int4(X); wq, ws = quant_int4(W)
    y = w4a4_gemm_v2(xq, xs, wq, ws)
    y64 = dequant_int4(xq, xs).double() @ dequant_int4(wq, ws).double().t()
    d = (y.double() - y64).abs().max().item()
    scale = max(1.0, y64.abs().max().item())
    print(f"   {M}x{N}x{K}: v2离真值={d:.4f} (|Y|={scale:.1f})")
    assert d < 0.05 * scale, (M, N, K)

print("\n== 最终性能总表 (4096^3 与 2048x512x4096) ==")
for (M, N, K) in [(2048, 512, 4096), (4096, 4096, 4096)]:
    X = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    W = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    X_cal = torch.randn(256, K, dtype=torch.float32, device='cuda')
    T = compute_wush_transform(X_cal, W.float(), group_size=32).to(torch.bfloat16)
    W_w = transform_weights(W.double(), T.double()).to(torch.bfloat16).contiguous()
    xqw, xsw = fused_transform_quant_int4_wmma(X, T)
    wqw, wsw = quant_int4(W_w)
    X16, W16 = X.half(), W.half()
    reps = 10

    t_fp16 = cuda_time(lambda: X16 @ W16.t(), reps=reps)
    t_v1 = cuda_time(lambda: w4a4_gemm_v1(xqw, xsw, wqw, wsw), reps=reps)
    t_v2 = cuda_time(lambda: w4a4_gemm_v2(xqw, xsw, wqw, wsw), reps=reps)
    t_fuse = cuda_time(lambda: fused_transform_quant_int4_wmma(X, T), reps=reps)
    t_e2e = t_fuse + t_v2

    print(f"\n   [{M}x{N}x{K}]")
    print(f"   fp16 cuBLAS GEMM          : {t_fp16:8.3f} ms          (基线)")
    print(f"   W4A4 GEMM v1 (逐组scale)  : {t_v1:8.3f} ms")
    print(f"   W4A4 GEMM v2 (逐组scale)  : {t_v2:8.3f} ms  (v1的 {t_v1/t_v2:.1f}x)")
    print(f"   融合变换+量化 (WMMA)       : {t_fuse:8.3f} ms")
    print(f"   ────────────────────────────────────")
    print(f"   INT4+WUSH 端到端          : {t_e2e:8.3f} ms  = fp16 的 {t_fp16/t_e2e:.2f}x")

print("\n== 端到端精度回归 (v2 全链路) ==")
M, N, K, GT = 2048, 512, 4096, 32
X = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
W = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
X[:, 5::GT] *= 15.0; W[:, 12::GT] *= 8.0
X_cal = torch.randn(256, K, dtype=torch.float32, device='cuda')
X_cal[:, 5::GT] *= 15.0
T = compute_wush_transform(X_cal, W.float(), group_size=GT)
W_w = transform_weights(W.double(), T.double()).to(torch.bfloat16).contiguous()
Tb = T.to(torch.bfloat16)
xqw, xsw = fused_transform_quant_int4_wmma(X, Tb)
wqw, wsw = quant_int4(W_w)
Y_true = X.double() @ W.double().t()
y = w4a4_gemm_v2(xqw, xsw, wqw, wsw)
e = ((y.double() - Y_true).norm() / Y_true.norm()).item()
print(f"   INT4+WUSH (融合WMMA + GEMM v2): {e:.4%}   (S5 基准 6.78%)")
assert e < 0.10

print("\nS-w4 收官完成")
