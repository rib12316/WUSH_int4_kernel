# sm_100 云端一键验证: 正确性 + 全形态 e2e vs fp16
# 用法: python run_sm100.py [--wmma]   (--wmma: GEMM 用 WMMA 路线替代 mma_sm75 原子)
import torch, sys, os, statistics, argparse
import os as _os, sys as _sys  # [P14] 仓库根上路径 (tests/bench/scripts 共用)
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))

parser = argparse.ArgumentParser()
parser.add_argument("--wmma", action="store_true", help="GEMM 走 WMMA 路线")
args = parser.parse_args()

from my_int4.rtn_int4 import quant_int4, dequant_int4
from my_int4.wush_transform import compute_wush_transform, transform_weights
from my_int4._C import (w4a4_gemm_s4, w4a4_gemm_v1, w4a4_gemv, w4a4_gemv_multi,
                        fused_transform_quant_int4_v3, fused_transform_quant_int4_smallm)
gemm_main = w4a4_gemm_v1 if args.wmma else w4a4_gemm_s4

assert torch.cuda.get_device_capability()[0] >= 10, "此脚本需 sm_100+ (5090/B200)"
print(f"GPU: {torch.cuda.get_device_name()} cc={torch.cuda.get_device_capability()}")

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

torch.manual_seed(0)
K = 4096
# ── 正确性 (fp64 仲裁) ──
print("== 正确性 ==")
for (M, N) in [(64, 128), (8, 4096), (2048, 512), (4096, 4096)]:
    X = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    W = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    xq, xs = quant_int4(X); wq, ws = quant_int4(W)
    y64 = dequant_int4(xq, xs).double() @ dequant_int4(wq, ws).double().t()
    d = (gemm_main(xq, xs, wq, ws).double() - y64).abs().max().item()
    ok = d < 0.05 * max(1.0, y64.abs().max().item())
    print(f"  {M}x{N}x{K}: {d:.4f} {'✓' if ok else '✗'}"); assert ok, (M, N)
# 精度 (outlier)
M, N = 2048, 512
X = torch.randn(M, K, dtype=torch.bfloat16, device='cuda'); X[:, 5::32] *= 15
W = torch.randn(N, K, dtype=torch.bfloat16, device='cuda'); W[:, 12::32] *= 8
Yt = X.double() @ W.double().t()
Xc = torch.randn(256, K, device='cuda'); Xc[:, 5::32] *= 15
T64 = compute_wush_transform(Xc, W.float(), group_size=32)
Ww = transform_weights(W.double(), T64).to(torch.bfloat16).contiguous()
wqw, wsw = quant_int4(Ww)
xqw, xsw = fused_transform_quant_int4_v3(X, T64.to(torch.bfloat16))
e = ((gemm_main(xqw, xsw, wqw, wsw).double() - Yt).norm() / Yt.norm()).item()
print(f"  outlier e2e 误差: {e:.4%} (本机基线 7.83%)"); assert e < 0.10

# ── 全形态 e2e (CUDA Graph) ──
print(f"\n== e2e vs fp16 (GEMM={'WMMA' if args.wmma else 's4 TC'}) ==")
print(f"{'M':>5} {'N':>5} | {'fp16':>8} | {'e2e':>8} {'加速比':>6}")
print("-" * 46)
res = {}
for (M, N) in [(1, 4096), (2, 4096), (4, 4096), (8, 4096), (32, 4096), (64, 4096),
               (2048, 512), (4096, 4096)]:
    X = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    W = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    X_cal = torch.randn(256, K, dtype=torch.float32, device='cuda')
    T64 = compute_wush_transform(X_cal, W.float(), group_size=32)
    T = T64.to(torch.bfloat16)
    W_w = transform_weights(W.double(), T64).to(torch.bfloat16).contiguous()
    wqw, wsw = quant_int4(W_w)
    X16, W16 = X.half(), W.half()
    reps = 100 if M <= 4 else (30 if M <= 64 else 8)
    t_fp16 = med5(lambda: X16 @ W16.t(), reps)
    if M == 1:
        def step():
            a, b = fused_transform_quant_int4_smallm(X, T)
            return w4a4_gemv(a, b, wqw, wsw)
    elif M <= 4:
        def step():
            a, b = fused_transform_quant_int4_smallm(X, T)
            return w4a4_gemv_multi(a, b, wqw, wsw)
    else:
        def step():
            a, b = fused_transform_quant_int4_v3(X, T)
            return gemm_main(a, b, wqw, wsw)
    try:
        g = graphed(step)
        t = med5(lambda: g.replay(), reps)
    except Exception:
        t = med5(step, reps)
    r = t_fp16 / t
    res[M] = r
    print(f"{M:>5} {N:>5} | {t_fp16:>7.3f}m | {t:>7.3f}m {r:>5.2f}x")
print()
ok = [m for m, r in res.items() if r >= 1.0]
print(f"≥1.0×: {sorted(ok)}  ({len(ok)}/8)")
print(f"<1.0×: {{{', '.join(f'{m}: {r:.2f}' for m, r in res.items() if r < 1.0)}}}")
print("\n达成则验收; 未达标请运行:")
print("  ncu --set full -k regex:'fused_tq|w4a4' -o report python run_sm100.py")
print("并将 report 与 NCU_FINDINGS.md 的 sm75 证据对照 (cp.async/TMA 应解除 MIO 墙)")
