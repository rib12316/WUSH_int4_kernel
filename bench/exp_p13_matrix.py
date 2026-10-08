# ============================================================================
# P13 基线/终态矩阵: GEMM 单核 + 9 形态 e2e (真 P11 分派), before/after 复用
# ============================================================================
import torch, sys, os, statistics
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

tag = sys.argv[1] if len(sys.argv) > 1 else "run"
torch.manual_seed(0)
K = 4096

print(f"== [{tag}] GEMM 单核 (graph med5, ms) ==")
for (M, N) in [(8, 4096), (64, 4096), (2048, 512), (4096, 4096)]:
    X = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    W = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    X_cal = torch.randn(256, K, dtype=torch.float32, device='cuda')
    T64 = compute_wush_transform(X_cal, W.float(), group_size=32)
    T = T64.to(torch.bfloat16)
    W_w = transform_weights(W.double(), T64).to(torch.bfloat16).contiguous()
    wqw, wsw = quant_int4(W_w)
    xqw, xsw = fused_transform_quant_int4_v3(X, T)
    reps = 100 if M <= 8 else (30 if M <= 64 else 8)
    g = graphed(lambda: w4a4_gemm_s4(xqw, xsw, wqw, wsw))
    t = med5(lambda: g.replay(), reps)
    print(f"   GEMM {M:>5}x{N:>5}x{K}: {t:8.4f} ms")
    del X, W, X_cal, T64, T, W_w, wqw, wsw, xqw, xsw, g
    torch.cuda.empty_cache()

print(f"\n== [{tag}] 9 形态 e2e (graph med5, vs fp16; 真 P11 分派) ==")
print(f"{'M':>5} {'N':>5} | {'fp16':>9} | {'e2e':>9} {'加速比':>6}")
print("-" * 46)
res = {}
for (M, N) in [(1, 4096), (2, 4096), (4, 4096), (8, 4096), (16, 4096), (32, 4096),
               (64, 4096), (2048, 512), (4096, 4096)]:
    X = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    W = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    X_cal = torch.randn(256, K, dtype=torch.float32, device='cuda')
    T64 = compute_wush_transform(X_cal, W.float(), group_size=32)
    T = T64.to(torch.bfloat16)
    W_w = transform_weights(W.double(), T64).to(torch.bfloat16).contiguous()
    wqw, wsw = quant_int4(W_w)
    X16, W16 = X.half(), W.half()
    reps = 100 if M <= 4 else (30 if M <= 64 else 8)
    # 真 P13 终态分派: M=1 smallm+gemv / M<=4 smallm+gemv_multi /
    # M<=32 smallm+gemm_s4(splitk) / M>=64 v3.1+gemm_s4 (实测链更快: 输出布局利 GEMM)
    if M == 1:
        def step(): a, b = fused_transform_quant_int4_smallm(X, T); return w4a4_gemv(a, b, wqw, wsw)
    elif M <= 4:
        def step(): a, b = fused_transform_quant_int4_smallm(X, T); return w4a4_gemv_multi(a, b, wqw, wsw)
    elif M <= 32:
        def step(): a, b = fused_transform_quant_int4_smallm(X, T); return w4a4_gemm_s4(a, b, wqw, wsw)
    else:
        def step(): a, b = fused_transform_quant_int4_v3(X, T); return w4a4_gemm_s4(a, b, wqw, wsw)
    t_fp16 = med5(lambda: X16 @ W16.t(), reps)
    g = graphed(step)                      # 捕获失败 = 直接异常 (哨兵)
    t = med5(lambda: g.replay(), reps)
    r = t_fp16 / t
    res[(M, N)] = r
    print(f"{M:>5} {N:>5} | {t_fp16:>7.3f}ms | {t:>7.3f}ms {r:>5.2f}x")
    del X, W, X_cal, T64, T, W_w, wqw, wsw, X16, W16, g
    torch.cuda.empty_cache()

ok = [k for k, v in res.items() if v >= 1.0]
print(f"\n[{tag}] ≥1.0×: {len(ok)}/9  {sorted(ok)}")
print(f"[{tag}] <1.0×: {{{', '.join(f'{m}:{r:.2f}' for (m, n), r in res.items() if r < 1.0)}}}")
