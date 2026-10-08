# S3 对拍: CUDA W4A4 GEMM vs S1 参考 —— 容差对拍 + fp64 第三方仲裁
# 口径(实现者决策): q 值两侧完全相同, 但 fp32 累加顺序不同(参考=逐元素反量化后
# cuBLAS; kernel=组内整数点积+组间FMA) → 原理性地不可能逐位一致(第5课"累加顺序"
# 教训)。所以: ①双方各自对 fp64 真值的偏差都要小(证明"都对") ②双方互相接近。
import torch, sys, os
import os as _os, sys as _sys  # [P14] 仓库根上路径 (tests/bench/scripts 共用)
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from my_int4.rtn_int4 import quant_int4, dequant_int4, w4a4_matmul_reference
from my_int4._C import quant_int4, w4a4_gemm

def check(X, W, name):
    xq, xs = quant_int4(X)          # CUDA 量化 (S2)
    wq, ws = quant_int4(W)
    y_cuda = w4a4_gemm(xq, xs, wq, ws)                       # (M, N) fp16
    y_ref = w4a4_matmul_reference(xq, xs, wq, ws)            # S1 参考 fp16

    # fp64 真值: 反量化的精确值做 fp64 矩阵乘
    xd = dequant_int4(xq, xs).double()
    wd = dequant_int4(wq, ws).double()
    y64 = xd @ wd.t()

    d_cuda = (y_cuda.double() - y64).abs().max().item()
    d_ref = (y_ref.double() - y64).abs().max().item()
    d_between = (y_cuda.float() - y_ref.float()).abs().max().item()
    scale = y64.abs().max().item()
    print(f"   {name}: |Y|={scale:.2f}  kernel离真值={d_cuda:.4f}  参考离真值={d_ref:.4f}  互相差={d_between:.4f}")
    assert d_cuda < 0.05 * max(1.0, scale), (name, "kernel 偏离 fp64 真值过大")
    assert d_ref < 0.05 * max(1.0, scale), (name, "参考偏离 fp64 真值过大")
    assert d_between < 0.05 * max(1.0, scale), (name, "kernel 与参考互相偏离过大")

torch.manual_seed(0)
print("== S3 对拍: CUDA W4A4 GEMM ==")
check(torch.randn(64, 128, dtype=torch.bfloat16, device='cuda'),
      torch.randn(96, 128, dtype=torch.bfloat16, device='cuda'), "常规 64x96x128")
check(torch.randn(1, 128, dtype=torch.bfloat16, device='cuda'),
      torch.randn(64, 128, dtype=torch.bfloat16, device='cuda'), "M=1 (decode形态)")
check(torch.randn(257, 96, dtype=torch.bfloat16, device='cuda'),
      torch.randn(33, 96, dtype=torch.bfloat16, device='cuda'), "非整块 M/N, K=96(3组)")
check(torch.randn(8, 32, dtype=torch.bfloat16, device='cuda'),
      torch.randn(8, 32, dtype=torch.bfloat16, device='cuda'), "K=32 单组")

# 边界: 全0 → 输出全0
X = torch.zeros(4, 64, dtype=torch.bfloat16, device='cuda')
W = torch.randn(16, 64, dtype=torch.bfloat16, device='cuda')
xq, xs = quant_int4(X); wq, ws = quant_int4(W)
y = w4a4_gemm(xq, xs, wq, ws)
assert (y == 0).all()
print("   全0激活: 输出全0 ✓")

print("\nS3 全部通过 — CUDA W4A4 GEMM v0 与参考实现一致(容差内), 且双方对 fp64 真值负责")
