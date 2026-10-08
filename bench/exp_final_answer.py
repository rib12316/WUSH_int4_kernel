# 决定性实验: WUSH变换 + 张量级scale INT4 (CUTLASS快路) 的精度 —— 变换是否
# 让"张量级 scale"逼近"逐组 scale"? 若是, 则 [融合变换量化 + CUTLASS GEMM]
# = 精度与速度兼得的最终推荐配置
import torch, sys, os
import os as _os, sys as _sys  # [P14] 仓库根上路径 (tests/bench/scripts 共用)
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from my_int4.wush_transform import compute_wush_transform, transform_weights
from my_int4._C import fused_transform_quant_int4_wmma, w4a4_gemm_v2, w4a4_gemm_cutlass

def cuda_time(fn, reps=20, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / reps

torch.manual_seed(0)
M, N, K, GT = 2048, 512, 4096, 32
X = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
W = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
X[:, 5::GT] *= 15.0; W[:, 12::GT] *= 8.0
Y_true = X.double() @ W.double().t()
X_cal = torch.randn(256, K, dtype=torch.float32, device='cuda'); X_cal[:, 5::GT] *= 15.0

# ---- 路线A: 逐组scale (v2 GEMM) —— 已知 6.90% ----
T64 = compute_wush_transform(X_cal, W.float(), group_size=GT)
T = T64.to(torch.bfloat16)
W_w = transform_weights(W.double(), T64).to(torch.bfloat16).contiguous()
xq, xs = fused_transform_quant_int4_wmma(X, T)

def quant_tensor(t):
    amax = t.float().abs().max().clamp_min(1e-8)
    s = (amax / 7.0).item()
    q = torch.clamp(torch.round(t.float() / s), -8, 7).to(torch.int8)
    return ((q[..., 0::2] & 0xF) | (q[..., 1::2] & 0xF) << 4).to(torch.uint8).contiguous(), s

# ---- 路线B: 变换 + 张量级scale (CUTLASS 快路) ----
# 融合kernel是逐组scale的 —— 为隔离"scale粒度"这单一变量, B 的激活也用
# torch 量化变换后的 X(张量级), 与权重的张量级量化对称
Xt = torch.einsum('bgi,gni->bgn', X.float().reshape(-1, K//GT, GT), T64.float()).flatten(-2).to(torch.bfloat16)
xqt, sxt = quant_tensor(Xt)
wqt, swt = quant_tensor(W_w)
y_b = w4a4_gemm_cutlass(xqt, wqt, sxt * swt)
e_b = ((y_b.double() - Y_true).norm() / Y_true.norm()).item()

# 参照: 不变换 + 张量级scale (S4c 原始形态)
xq0, sx0 = quant_tensor(X); wq0, sw0 = quant_tensor(W)
y0 = w4a4_gemm_cutlass(xq0, wq0, sx0 * sw0)
e0 = ((y0.double() - Y_true).norm() / Y_true.norm()).item()

print("== 精度: scale 粒度 × 是否变换 (outlier 数据) ==")
print(f"  不变换 + 张量级scale   : {e0:>8.4%}")
print(f"  WUSH变换 + 张量级scale : {e_b:>8.4%}")
print(f"  WUSH变换 + 逐组scale   :   6.8974%  (已知, 路线A)")
print(f"  → 变换把张量级scale的差距缩到: {e_b/0.0690:.2f}x")

# ---- 性能: 推荐配置的端到端 ----
xqw, xsw = fused_transform_quant_int4_wmma(X, T)
wqw_t, swt2 = quant_tensor(W_w)
t_fuse = cuda_time(lambda: fused_transform_quant_int4_wmma(X, T))
t_cl = cuda_time(lambda: w4a4_gemm_cutlass(xqt, wqt, sxt*swt))
t_v2 = cuda_time(lambda: w4a4_gemm_v2(xqw, xsw, wqw_t, torch.full_like(xsw, swt2)))
X16, W16 = X.half(), W.half()
t_fp16 = cuda_time(lambda: X16 @ W16.t())
print(f"\n== 性能 [{M}x{N}x{K}] ==")
print(f"  fp16 cuBLAS                     : {t_fp16:7.3f} ms   (基线)")
print(f"  A: 融合(逐组) + GEMM v2(逐组)    : {t_fuse+t_v2:7.3f} ms  = fp16 的 {t_fp16/(t_fuse+t_v2):.2f}x")
print(f"  B: 融合(逐组) + CUTLASS(张量级)  : {t_fuse+t_cl:7.3f} ms  = fp16 的 {t_fp16/(t_fuse+t_cl):.2f}x")
