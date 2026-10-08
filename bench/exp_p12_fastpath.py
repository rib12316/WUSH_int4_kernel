# ============================================================================
# P12 快速路线基准: amax→quant(per-tensor)→CUTLASS s4, 全形态 vs fp16
# ============================================================================
# 机制逐字沿用 run_sm100.py (cuda_time/med5/graphed), 保证与 P10/P11 终态数字同口径:
#   - fp16 基线: eager med5 (与历史发布口径一致)
#   - fast-eager: 含 amax .item() host 同步 (如实报告 —— 它量化了 graph 化消掉了什么)
#   - fast-graph: 捕获后 med5(g.replay()); 捕获失败直接异常, 不静默回退
# 9 形态 = run_sm100 的 8 个 + M=16 (与 P10/P11 表衔接)
import torch, sys, os, statistics
import os as _os, sys as _sys  # [P14] 仓库根上路径 (tests/bench/scripts 共用)
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from my_int4 import _C

assert torch.cuda.get_device_capability() == (7, 5), "此基准锚定本机 RTX 2060 (sm75)"
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

def amax_scale(t):
    amax = t.float().abs().max().clamp_min(1e-8)
    return (amax / 7.0).item(), amax

def fast_step(X, wqt, sw_host):
    """全 GPU 快速路线链 (可捕获): amax→inv→量化→CUTLASS(alpha_ptr)"""
    amax = X.abs().max().float().clamp_(min=1e-8)
    inv = 7.0 / amax
    xq = _C.quant_int4_tensor_scale_dev(X, inv)
    alpha = amax * (sw_host / 7.0)
    return _C.w4a4_gemm_cutlass_dev(xq, wqt, alpha)

K = 4096
torch.manual_seed(0)

# ── A. 量化 kernel 单核带宽 (信息项: 距带宽地板多远) ──
print("== A. quant_int4_tensor_scale 单核 (2.5B/元素: 2B 读 + 0.5B 写) ==")
for (M, NK) in [(64, 4096), (2048, 512), (4096, 4096)]:
    X = torch.randn(M, K, dtype=torch.bfloat16, device='cuda').contiguous()
    sx, _ = amax_scale(X)
    reps = 30 if M <= 64 else 8
    t = med5(lambda: _C.quant_int4_tensor_scale(X, sx), reps)
    gbs = 2.5 * X.numel() / (t * 1e-3) / 1e9
    print(f"   ({M:>4},{K})→{X.numel()/1e6:>5.1f}M 元素: {t*1e3:>8.1f} µs   {gbs:6.0f} GB/s")

# ── B. e2e 全形态表: fp16 / fast-eager / fast-graph ──
print("\n== B. e2e vs fp16 (K=4096) ==")
print(f"{'M':>5} {'N':>5} | {'fp16':>9} | {'eager':>9} {'比':>6} | {'graph':>9} {'比':>6}")
print("-" * 64)
res = {}
for (M, N) in [(1, 4096), (2, 4096), (4, 4096), (8, 4096), (16, 4096), (32, 4096),
               (64, 4096), (2048, 512), (4096, 4096)]:
    X = torch.randn(M, K, dtype=torch.bfloat16, device='cuda').contiguous()
    W = torch.randn(N, K, dtype=torch.bfloat16, device='cuda').contiguous()
    sw, _ = amax_scale(W)
    wqt = _C.quant_int4_tensor_scale(W, sw).view(N, -1)
    X16, W16 = X.half(), W.half()
    reps = 100 if M <= 4 else (30 if M <= 64 else 8)

    def eager_step():                       # host scale: amax .item() 同步每步发生
        sx = (X.float().abs().max().clamp_min(1e-8) / 7.0).item()
        xq = _C.quant_int4_tensor_scale(X, sx)
        return _C.w4a4_gemm_cutlass(xq, wqt, sx * sw)

    t_fp16 = med5(lambda: X16 @ W16.t(), reps)
    t_eager = med5(eager_step, reps)
    g = graphed(lambda: fast_step(X, wqt, sw))          # 捕获失败 = 直接异常 (哨兵)
    t_graph = med5(lambda: g.replay(), reps)
    r_e, r_g = t_fp16 / t_eager, t_fp16 / t_graph
    res[(M, N)] = (r_e, r_g)
    print(f"{M:>5} {N:>5} | {t_fp16:>7.3f}ms | {t_eager:>7.3f}ms {r_e:>5.2f}x | {t_graph:>7.3f}ms {r_g:>5.2f}x")

# ── C. 分派建议: fast-graph vs 逐组路线终态 (P11) ──
prior = {(1, 4096): 2.75, (2, 4096): 2.07, (4, 4096): 1.19, (8, 4096): 0.81,
         (16, 4096): 1.17, (32, 4096): 1.36, (64, 4096): 0.88,
         (2048, 512): 0.46, (4096, 4096): 0.85}   # P11 逐组路线 e2e 终态
print("\n== C. 分派建议 (每形态取二者较优) ==")
print(f"{'形态':>12} | {'fast-graph':>10} | {'逐组(P11)':>9} | 建议")
for shape, (r_e, r_g) in res.items():
    p = prior[shape]
    pick = "fast" if r_g >= max(1.0, p) else ("逐组" if p >= r_g else "fast")
    best = max(r_g, p)
    tag = "" if best >= 1.0 else " (均<1.0)"
    print(f"{str(shape):>12} | {r_g:>9.2f}x | {p:>8.2f}x | {pick} ({best:.2f}x){tag}")

ok = [s for s, (_, r_g) in res.items() if r_g >= 1.0]
print(f"\nfast-graph ≥1.0×: {len(ok)}/{len(res)} 形态: {sorted(ok)}")

# ── D. 信息项: outlier 精度 (不计精度决策的代价实录) ──
M, N, GT = 2048, 512, 32
X = torch.randn(M, K, dtype=torch.bfloat16, device='cuda'); X[:, 5::GT] *= 15.0
W = torch.randn(N, K, dtype=torch.bfloat16, device='cuda'); W[:, 12::GT] *= 8.0
Yt = X.double() @ W.double().t()
sx, _ = amax_scale(X); sw2, _ = amax_scale(W)
xq = _C.quant_int4_tensor_scale(X, sx); wq = _C.quant_int4_tensor_scale(W, sw2).view(N, -1)
e = ((_C.w4a4_gemm_cutlass(xq, wq, sx * sw2).double() - Yt).norm() / Yt.norm()).item()
print(f"\n== D. outlier e2e 误差 (信息项): {e:.2%}  (逐组路线 7.83%; P12 决策不计精度) ==")
