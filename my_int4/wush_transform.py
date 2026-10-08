# ============================================================================
# S-w1: WUSH 变换的离线管线 (Python) —— 求解 T / 变换权重 / 激活变换参考
# ============================================================================
# 数学全部复用 WUSH 的 block_transform.py (第3课跑通过它的单元测试),
# 本文件只做工程封装并钉死三件事:
#   1. T 从哪来: 校准激活 X_cal 与权重 W 的分块二阶矩 → 闭式解 (block_transform.py:168)
#   2. 权重怎么变: W' = W @ T^{-1} (每 block; 等价写法 apply(W, T^{-T}, inv=False))
#      —— 方向不能错, 配对关系见 SPEC/第3课恒等式: X'T^T·(W'T^{-1})^T = XW^T
#   3. 激活怎么变: X' = X @ T^T (每 block) —— 这是 S-w2 融合 kernel 的语义基准
import sys, os

# WUSH 上游依赖解析 (block_transform / compute_gram 来自 WUSH 仓库 src/):
#   默认取本仓库的兄弟目录 ../WUSH; 可用环境变量 WUSH_ROOT 覆盖。
#   [P14 重构] 路径逻辑收拢于此 —— 各测试/基准脚本不再各自 sys.path hack。
_WUSH_ROOT = os.environ.get(
    "WUSH_ROOT",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "WUSH")))
_WUSH_SRC = os.path.join(_WUSH_ROOT, "src")
if _WUSH_SRC not in sys.path:
    sys.path.insert(0, _WUSH_SRC)

import torch
import block_transform as bt
from compute_gram import get_diag_block_gram


def compute_wush_transform(X_cal: torch.Tensor, W: torch.Tensor,
                           group_size: int = 32, use_hadamard: bool = True) -> torch.Tensor:
    """由校准数据求解每 block 的 WUSH 变换矩阵。
    X_cal: (M, K) 校准激活; W: (N, K) 权重
    返回: T (K/GT, GT, GT) fp64 —— kernel 侧再转 bf16/fp16"""
    hi = torch.float64
    M, K = X_cal.shape
    N = W.shape[0]
    # 分块二阶矩 (Gram): 对角块各算各的 —— "per-block" 的数据含义就在这
    G_x = get_diag_block_gram(x=X_cal, size=group_size, dtype=hi) / M     # (C, GT, GT)
    G_w = get_diag_block_gram(x=W,     size=group_size, dtype=hi) / N
    return bt.get_wush_transform(gram_weight=G_w, gram_activation=G_x,
                                 use_hadamard=use_hadamard,
                                 high_dtype=hi, round_dtype=hi)


def transform_weights(W: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """离线(部署前一次): W' = W @ T^{-1} (每 block)。输入输出均可 fp64/bf16。"""
    hi = torch.float64
    T_w = bt.get_inversed_transposed_transform(T.to(hi), high_dtype=hi, round_dtype=hi)
    # apply(x, transform, inv=False) = x @ transform^T (每 block)
    # transform = T^{-T}  →  W' = W @ (T^{-T})^T = W @ T^{-1}   ✓ 配对方向
    return bt.apply_block_transform(x=W.to(hi), transform=T_w.to(hi),
                                    is_inverse_transpose=False,
                                    high_dtype=hi, round_dtype=hi)


def transform_activations_reference(X: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """激活变换的参考语义 (S-w2 CUDA kernel 的对拍真值): X' = X @ T^T (每 block)"""
    return bt.apply_block_transform(x=X.to(torch.float64), transform=T,
                                    is_inverse_transpose=False,
                                    high_dtype=torch.float64, round_dtype=torch.float64)
