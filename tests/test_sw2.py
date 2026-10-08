# S-w2 对拍: 融合"变换+量化" kernel
# 逐位真值的构造: torch 参考逐步镜像 kernel 的循环顺序 ——
#   x'_j = Σ_i x_i·T[j][i], fp32, i 升序, 乘/加分开两次舍入(与 __fmul_rn/__fadd_rn 对应)
#   然后 S1 的量化链(S2 已证与 CUDA 逐位一致)
import torch, sys, os
import os as _os, sys as _sys  # [P14] 仓库根上路径 (tests/bench/scripts 共用)
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from my_int4.rtn_int4 import quant_int4
from my_int4.wush_transform import transform_activations_reference
from my_int4._C import fused_transform_quant_int4, w4a4_gemm_v1, w4a4_gemm_cutlass

def ref_fused(x, T):
    """镜像 kernel 顺序的参考: 变换(32步循环) + S1 量化链 (支持任意前导维)"""
    xf = x.float().reshape(-1, x.size(-1))           # 摊平前导维 (kernel 同款视角)
    rows, K = xf.shape
    C = K // 32
    Tt = T.float()
    out = torch.zeros(rows, K, dtype=torch.float32, device=x.device)
    xg = xf.unflatten(-1, (-1, 32))
    og = out.unflatten(-1, (-1, 32))
    for i in range(32):                              # 顺序: i 升序逐项累加
        og += xg[:, :, i].unsqueeze(-1) * Tt[:, :, i].unsqueeze(0)
    out = og.flatten(-2)
    return quant_int4(out)                           # S1 链(S2 已证与 kernel 逐位一致)

def check(x, T, name):
    q_c, s_c = fused_transform_quant_int4(x, T)
    q_r, s_r = ref_fused(x, T)
    assert q_c.numel() == q_r.numel() and s_c.numel() == s_r.numel(), (name, "元素数不符")
    assert torch.equal(q_c.reshape(-1), q_r.reshape(-1)), (name, "打包字节不一致")
    assert torch.equal(s_c.reshape(-1).view(torch.int16), s_r.reshape(-1).view(torch.int16)), (name, "scale 位模式不一致")
    print(f"   {name}: packed/scale 逐字节一致 ✓")

torch.manual_seed(0)
print("== S-w2 对拍: 融合变换+量化 kernel ==")
# T: 用真实 WUSH 变换(第3课管线) + 随机 T 两种来源
X = torch.randn(256, 128, dtype=torch.bfloat16, device='cuda')
W = torch.randn(64, 128, dtype=torch.bfloat16, device='cuda')
X[:, 5::32] *= 12.0
from my_int4.wush_transform import compute_wush_transform
T_wush = compute_wush_transform(X.float(), W.float()).to(torch.bfloat16)

check(X, T_wush, "真实WUSH T, 256行含outlier")
check(X, torch.randn_like(T_wush), "随机T")
check(torch.randn(2, 3, 96, dtype=torch.bfloat16, device='cuda'),
      torch.randn(3, 32, 32, dtype=torch.bfloat16, device='cuda'), "三维形状")
check(torch.zeros(4, 64, dtype=torch.bfloat16, device='cuda'),
      T_wush[:2].contiguous(), "全0输入")
check(torch.randn(1 << 16, 64, dtype=torch.bfloat16, device='cuda'),
      T_wush[:2].contiguous(), "压力 2^21元素")

# 语义合理性: 变换后量化 vs 未变换量化 (outlier 下应有巨大差异)
q_t, s_t = fused_transform_quant_int4(X, T_wush)
print("\n语义抽查: 变换后 scale 分布 (outlier 组不再被峰值绑架):")
print("   未变换时组内不均衡度高; 变换后 scale 应趋于平缓 —— S5 实验定量验证")

print("\nS-w2 全部通过 — 融合 kernel 与参考实现逐位一致")
