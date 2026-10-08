# ============================================================================
# P13: gemm_s4 half2 scale 流 (stage-flush) 正确性闸门
# ============================================================================
# ① fp64 仲裁: w4a4_gemm_s4 vs 反量化 fp64 真值, 5 形态覆盖全分派分支
#    (splitk TM32+S4 / splitk TM64+S2 / M=1 / 非整块 / 多tile 大K)
# ② outlier e2e 硬闸: WUSH 链 (v3 融合 + s4 GEMM) 相对误差 ≤ 8.2%
#    (基线 7.83%; stage-flush 数值路径的代价实录)
import torch, sys, os
import os as _os, sys as _sys  # [P14] 仓库根上路径 (tests/bench/scripts 共用)
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from my_int4.rtn_int4 import quant_int4, dequant_int4
from my_int4.wush_transform import compute_wush_transform, transform_weights
from my_int4 import _C

torch.manual_seed(0)

print("== P13-1 fp64 仲裁 (half2 stage-flush 路径) ==")
for (M, N, K) in [(64, 96, 128), (1, 64, 64), (300, 500, 512), (130, 512, 4096), (8, 4096, 4096)]:
    X = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    W = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    xq, xs = quant_int4(X); wq, ws = quant_int4(W)
    y = _C.w4a4_gemm_s4(xq, xs, wq, ws)
    y64 = dequant_int4(xq, xs).double() @ dequant_int4(wq, ws).double().t()
    d = (y.double() - y64).abs().max().item()
    tol = 0.05 * max(1.0, y64.abs().max().item())
    print(f"   {M}x{N}x{K}: 离真值={d:.4f} (|Y|={y64.abs().max().item():.1f})")
    assert d < tol, (M, N, K, d, tol)

print("\n== P13-2 outlier e2e 硬闸 (v3 融合 + s4 GEMM, ≤8.2%) ==")
M, N, K, GT = 2048, 512, 4096, 32
X = torch.randn(M, K, dtype=torch.bfloat16, device='cuda'); X[:, 5::GT] *= 15.0
W = torch.randn(N, K, dtype=torch.bfloat16, device='cuda'); W[:, 12::GT] *= 8.0
Yt = X.double() @ W.double().t()
Xc = torch.randn(256, K, device='cuda'); Xc[:, 5::GT] *= 15.0
T64 = compute_wush_transform(Xc, W.float(), group_size=GT)
Ww = transform_weights(W.double(), T64).to(torch.bfloat16).contiguous()
wqw, wsw = quant_int4(Ww)
xqw, xsw = _C.fused_transform_quant_int4_v3(X, T64.to(torch.bfloat16))
e = ((_C.w4a4_gemm_s4(xqw, xsw, wqw, wsw).double() - Yt).norm() / Yt.norm()).item()
print(f"   outlier e2e 误差: {e:.4%}  (P11 基线 7.83%, 闸门 8.2%)")
if e > 8.0:
    print("   [早警] 已 >8.0%, 接近闸门")
assert e <= 0.082, e

print("\n== P13-3 graph 捕获 sanity (M=8 splitk 原子链) ==")
M, N, K = 8, 4096, 4096
X = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
W = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
Xc = torch.randn(256, K, device='cuda')
T64 = compute_wush_transform(Xc, W.float(), group_size=32)
T = T64.to(torch.bfloat16)
Ww = transform_weights(W.double(), T64).to(torch.bfloat16).contiguous()
wqw, wsw = quant_int4(Ww)
def step():
    a, b = _C.fused_transform_quant_int4_smallm(X, T)
    return _C.w4a4_gemm_s4(a, b, wqw, wsw)
y0 = step().clone()                     # eager 基准 (capture 上下文的张量未执行, 不可作基准)
for _ in range(3): step()
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): out = step()
g.replay(); torch.cuda.synchronize()
# [P13b] splitk 原子直写的加法序不定 → 位一致不成立, 用容差 (S=4 个 fp16 原子和)
d = (out.float() - y0.float()).abs().max().item()
print(f"   M=8 splitk 原子链 graph replay 最大偏差: {d:.4f} (容差 0.5)")
assert d < 0.5, d

print("\nP13 测试通过")
