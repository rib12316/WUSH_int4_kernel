# S2 对拍: CUDA kernel vs S1 参考实现 —— 逐字节一致才算过
import torch, sys, os
import os as _os, sys as _sys  # [P14] 仓库根上路径 (tests/bench/scripts 共用)
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from my_int4.rtn_int4 import quant_int4 as ref_quant
from my_int4._C import quant_int4 as cuda_quant

def check(x, name):
    q_ref, s_ref = ref_quant(x)
    q_cuda, s_cuda = cuda_quant(x)
    assert q_cuda.shape == q_ref.shape, (name, q_cuda.shape, q_ref.shape)
    assert s_cuda.numel() == s_ref.numel(), (name, s_cuda.shape, s_ref.shape)
    assert torch.equal(q_cuda, q_ref), (name, f"FP4打包字节不一致, 首个分歧@{(q_cuda != q_ref).nonzero()[0].tolist()}")
    # scale 用位模式比较(绕开 NaN/语义比较的坑, 严格到 bit 级)
    assert torch.equal(s_cuda.reshape(-1).view(torch.int16), s_ref.reshape(-1).view(torch.int16)), (name, "scale 位模式不一致")
    print(f"   {name}: packed/scale 逐字节一致 ✓")

torch.manual_seed(0)
print("== S2 对拍: CUDA kernel vs rtn_int4 ==")
check(torch.randn(4096, 32, dtype=torch.bfloat16, device='cuda'), "常规随机 4096组")
check(torch.randn(2, 3, 96, dtype=torch.bfloat16, device='cuda'), "三维形状")
check(torch.zeros(4, 32, dtype=torch.bfloat16, device='cuda'), "全0组")
check(torch.randn(1 << 17, 128, dtype=torch.bfloat16, device='cuda'), "压力 2^24元素")

# 平局值: amax=7 → scale=1 → x/scale 落在半整数上
x = torch.zeros(2, 32, dtype=torch.bfloat16, device='cuda')
x[0, :8] = torch.tensor([7., -7., 2.5, -2.5, 3.5, -3.5, 0.5, -0.5])
x[1, :8] = torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, -6.5]) * 0.7   # amax=4.55 → 非整数scale的平局
check(x, "平局值组")

# 大数: 溢出路径(Σx²=inf → var=NaN → scale回退1.0, q饱和到±7)
x = torch.full((4, 32), 3.3e38, dtype=torch.bfloat16, device='cuda')
q, s = cuda_quant(x)
q_ref, s_ref = ref_quant(x)
assert torch.equal(q, q_ref) and torch.equal(s.reshape(-1).view(torch.int16), s_ref.reshape(-1).view(torch.int16))
print("   大数溢出路径: 与参考逐字节一致 ✓")

print("\nS2 全部通过 — CUDA 量化 kernel 与参考实现逐位一致")
