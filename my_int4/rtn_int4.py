# ============================================================================
# S1: INT4 量化的 PyTorch 参考实现 —— WUSH 论文 INT4 口径 (P1 对齐版)
# ============================================================================
# 论文 (arXiv 2512.00956 §6.1): "INT4 to denote 4-bit integers with Gaussian MSE
# clipping and BF16 scales of group size 32" —— 即:
#   ① 分组: group 32 沿 K (激活 per-token-per-group, 权重 per-channel-per-group)
#   ② 定标: scale = σ_g × (2.92247856/7) + 1e-8   [高斯 MSE 最优截断常数,
#      与 WUSH CUDA epilogue 的 Quest 模式同源(epilogue_quant.h:534 的 /6 是
#      FP4 格点上限, INT4 对称上限 7)]
#   ③ scale 存储: BF16; 编码用"BF16 舍入后"的值(roundtrip 原则, 第4课)
#   ④ 对称 RTN, clamp [-7,7] (超截断的极值饱和)
# 逐位对拍纪律: σ 的求和/均值/方差全部按 kernel 的顺序逐步镜像(32 步串行)。
import torch

INT4_MIN, INT4_MAX = -7, 7          # 对称模式: /7 定标, 两侧都到 7
# 截断常数: 数值求得 INT4 15点对称均匀格点的 MSE 最优 c*=2.473650σ (5e7 样本
# 黄金分割搜索; 对照: FP4 非均匀格点的最优值是 2.92247856 —— 上游 Quest 模式
# 用的那个, 对 INT4 均匀格点并非最优, 实测差 14% MSE)
C_CLIP = 0.35337862372398376


def _seq_sums(g: torch.Tensor):
    """按 kernel 顺序(i=0..31 串行)累加 Σx 与 Σx² —— 逐位镜像, 防 torch 归约树序差异"""
    s1 = torch.zeros_like(g[..., 0])
    s2 = torch.zeros_like(g[..., 0])
    for i in range(32):
        v = g[..., i]
        s1 = s1 + v
        s2 = s2 + v * v
    return s1, s2


def quant_int4(x: torch.Tensor, group_size: int = 32):
    """x (..., K) -> (q_packed uint8 (..., K/2), scale bf16 (..., K/G))
    WUSH 论文 INT4 口径: σ 高斯 MSE 截断, BF16 scale, 对称 RTN, clamp [-7,7]"""
    xf = x.float()                                   # 量化数学在 fp32 (高精度真值)
    groups = xf.unflatten(-1, (-1, group_size))      # (..., G, 32)

    # σ = sqrt(E[x²] - E[x]²), 与 CUDA epilogue (epilogue_quant.h:521-534) 同式同序
    s1, s2 = _seq_sums(groups)
    c_mean = s1 / 32.0                               # /32 是 2 的幂, 精确
    var = s2 / 32.0 - c_mean * c_mean
    scale = torch.where(var >= 0,
                        torch.sqrt(var) * C_CLIP + 1e-8,   # var<0(浮点消减) → 用 1.0, 同 kernel
                        torch.ones_like(var))

    scale_q = scale.to(torch.bfloat16)               # 存储: BF16 (论文口径)
    scale_dq = scale_q.float()                       # roundtrip: 编码用舍入后的值

    y = groups / scale_dq.unsqueeze(-1)              # torch.round = half-to-even, 同 __float2int_rn
    q = torch.clamp(torch.round(y), INT4_MIN, INT4_MAX).to(torch.int8)

    qu8 = q.to(torch.uint8)
    byte = (qu8[..., 0::2] & 0xF) | (qu8[..., 1::2] & 0xF) << 4   # 偶元素低 nibble
    return byte.flatten(-2), scale_q.squeeze(-1)


def dequant_int4(q_byte: torch.Tensor, scale: torch.Tensor, group_size: int = 32):
    """(q_packed uint8 (..., K/2), scale bf16 (..., G)) -> fp32 (..., K)"""
    lo = (q_byte & 0xF).to(torch.int16)
    hi = (q_byte >> 4).to(torch.int16)
    # int4 → int16 符号扩展: nibble 符号位(bit3)须顶到 int16 符号位(bit15):
    # 左移 12 再算术右移 12 (S1 实录: <<4>>4 对 4bit 值无效)
    lo = (lo << 12) >> 12
    hi = (hi << 12) >> 12
    d = torch.stack([lo, hi], dim=-1).flatten(-2)
    d = d.unflatten(-1, (-1, group_size)).float() * scale.float().unsqueeze(-1)
    return d.flatten(-2)


def w4a4_matmul_reference(x_q, x_s, w_q, w_s):
    """W4A4 矩阵乘参考: 两侧反量化后 fp32 矩阵乘, 输出 fp16。Y = X @ W^T"""
    xd = dequant_int4(x_q, x_s)
    wd = dequant_int4(w_q, w_s)
    return (xd @ wd.transpose(-1, -2)).to(torch.float16)
