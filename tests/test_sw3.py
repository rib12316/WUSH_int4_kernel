# S-w3 对拍 + 性能: 融合变换量化 v1 (WMMA)
# 口径: WMMA 的累加顺序与 v0 串行循环不同(fp32 最后几 ulp) → 少数平局码会翻转,
#       因此验收 = ①码翻转率极低 ②scale 一致 ③端到端误差与 v0 链路同水平
import torch, sys, os
import os as _os, sys as _sys  # [P14] 仓库根上路径 (tests/bench/scripts 共用)
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from my_int4.wush_transform import compute_wush_transform
from my_int4._C import quant_int4, w4a4_gemm_v1, fused_transform_quant_int4, fused_transform_quant_int4_wmma
from my_int4.wush_transform import transform_weights

torch.manual_seed(0)
M, N, K, GT = 2048, 512, 4096, 32
dev = torch.device('cuda')
X = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
W = torch.randn(N, K, dtype=torch.bfloat16, device=dev)
X[:, 5::GT] *= 15.0
W[:, 12::GT] *= 8.0
X_cal = torch.randn(256, K, dtype=torch.float32, device=dev)
X_cal[:, 5::GT] *= 15.0
T = compute_wush_transform(X_cal, W.float(), group_size=GT).to(torch.bfloat16)

print("== 正确性 ==")
q0, s0 = fused_transform_quant_int4(X, T)
q1, s1 = fused_transform_quant_int4_wmma(X, T)
n_flip = (q1 != q0).sum().item()
print(f"   码翻转率: {n_flip / q0.numel() * 100:.4f}%  (WMMA 与串行累加的最后几 ulp 差异, 仅平局码翻转)")
assert n_flip / q0.numel() < 0.02
assert torch.allclose(s1.float(), s0.float(), rtol=1e-2), "scale 不一致"

# 端到端精度: v1 融合 kernel 的完整链路 (对照 S5 的 6.78%)
W_w = transform_weights(W.double(), T.double()).to(torch.bfloat16).contiguous()
wqw, wsw = quant_int4(W_w)
xqw, xsw = fused_transform_quant_int4_wmma(X, T)
Y_true = X.double() @ W.double().t()
y = w4a4_gemm_v1(xqw, xsw, wqw, wsw)
e_wush = ((y.double() - Y_true).norm() / Y_true.norm()).item()
print(f"   端到端 W4A4+WUSH 误差 (v1融合kernel): {e_wush:.4%}  (S5 的 v0 版为 6.78%)")
assert e_wush < 0.10

print("\n== 性能 (ms) ==")
def cuda_time(fn, reps=30, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / reps

t_f0 = cuda_time(lambda: fused_transform_quant_int4(X, T), reps=5, warmup=2)
t_f1 = cuda_time(lambda: fused_transform_quant_int4_wmma(X, T), reps=30)  # 注意: 返回3元组, 计时不解包
t_q = cuda_time(lambda: quant_int4(X), reps=30)
gb = M * K * 2 / 1e9
print(f"   普通量化        : {t_q:>8.3f} ms ({gb/t_q*1e3:6.0f} GB/s)")
print(f"   融合v0 (串行)   : {t_f0:>8.3f} ms ({gb/t_f0*1e3:6.0f} GB/s)")
print(f"   融合v1 (WMMA)   : {t_f1:>8.3f} ms ({gb/t_f1*1e3:6.0f} GB/s)  加速 {t_f0/t_f1:.1f}x")

# 端到端总账
X16, W16 = X.half(), W.half()
t_fp16 = cuda_time(lambda: X16 @ W16.t(), reps=30)
t_gem = cuda_time(lambda: w4a4_gemm_v1(xqw, xsw, wqw, wsw), reps=30)
t_total = t_f1 + t_gem
print(f"\n   端到端 INT4+WUSH (v1融合 + v1 GEMM): {t_total:.3f} ms  vs fp16 GEMM {t_fp16:.3f} ms  = {t_fp16/t_total:.2f}x")
