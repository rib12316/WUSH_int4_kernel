# S-w1 对拍: 变换管线的三道闸门
import torch, sys, os
import os as _os, sys as _sys  # [P14] 仓库根上路径 (tests/bench/scripts 共用)
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from my_int4.wush_transform import compute_wush_transform, transform_weights, transform_activations_reference
from my_int4.rtn_int4 import quant_int4, w4a4_matmul_reference

dev = torch.device('cuda')
torch.manual_seed(0)
M, N, K, GT = 256, 128, 128, 32

# ---- 闸门1: 乘积不变性 (核心数学) ----
# X'W'^T 必须等于 XW^T —— WUSH 一切价值的根基
X64 = torch.randn(M, K, dtype=torch.float64, device=dev)
W64 = torch.randn(N, K, dtype=torch.float64, device=dev)
# 制造 outlier (让 WUSH 真的有活干)
X64[:, 5::GT] *= 15.0
W64[:, 12::GT] *= 8.0

T = compute_wush_transform(X64, W64, group_size=GT)            # (K/GT, GT, GT) fp64
Xp = transform_activations_reference(X64, T)
Wp = transform_weights(W64, T)
Y_true = X64 @ W64.t()
Y_new = Xp @ Wp.t()
rel = ((Y_new - Y_true).norm() / Y_true.norm()).item()
print(f"闸门1 乘积不变性: ||X'W'^T - XW^T||/||XW^T|| = {rel:.2e}  (fp64, 应≈1e-12)")
assert rel < 1e-9

# ---- 闸门2: 权重变换的两种等价写法 (第3课单元测试 :298-300 的代码面) ----
import block_transform as bt
T_w = bt.get_inversed_transposed_transform(T, high_dtype=torch.float64, round_dtype=torch.float64)
Wa = bt.apply_block_transform(x=W64, transform=T, is_inverse_transpose=True)     # W @ T^{-1}
Wb = bt.apply_block_transform(x=W64, transform=T_w, is_inverse_transpose=False)  # W @ T^{-1}
# 口径说明: 两条路径(solve vs inv+matmul)的浮点顺序不同, 官方单测也是 allclose
# (block_transform.py:300) —— 同一数学的不同路径只能要求容差内一致
diff = (Wa - Wb).abs().max().item()
assert torch.allclose(Wa, Wb, rtol=1e-9, atol=1e-9), (f"差 {diff}")
print(f"闸门2 两种等价写法: 最大差 {diff:.2e} (fp64 容差内一致) ✓")

# ---- 闸门3: 真实管线预览 (bf16 + 量化) —— S5 的预告片 ----
X = X64.to(torch.bfloat16)
W = W64.to(torch.bfloat16)
T_bf = T.to(torch.bfloat16)
Xp16 = transform_activations_reference(X, T_bf).to(torch.bfloat16)
Wp16 = transform_weights(W, T_bf).to(torch.bfloat16)

def w4a4_error(Xa, Wa):
    xq, xs = quant_int4(Xa)
    wq, ws = quant_int4(Wa)
    y = w4a4_matmul_reference(xq, xs, wq, ws).float()
    y_true = (Xa.float() @ Wa.float().t())
    return ((y - y_true).norm() / y_true.norm()).item()

e_bare = w4a4_error(X, W)          # 裸 INT4
e_wush = w4a4_error(Xp16, Wp16)    # INT4 + WUSH 变换 (权重离线变换 + 激活参考变换)
y_fp16 = (X.float() @ W.float().t())
e_fp16_base = ((X.half().float() @ W.half().float().t()) - y_fp16).norm().item() / y_fp16.norm().item()
print(f"闸门3 端到端误差 (outlier 数据): fp16基线={e_fp16_base:.4%}  裸INT4={e_bare:.4%}  INT4+WUSH={e_wush:.4%}")
print("   → S5 将用 CUDA 融合 kernel 替换这里的 Python 参考变换后复测")
