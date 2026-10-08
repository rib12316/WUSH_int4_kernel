import torch, sys
import os as _os, sys as _sys  # [P14] 仓库根上路径 (tests/bench/scripts 共用)
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from my_int4.rtn_int4 import quant_int4
from my_int4.wush_transform import compute_wush_transform, transform_weights
from my_int4._C import w4a4_gemm_s4, fused_transform_quant_int4_wmma
import sys
which = sys.argv[1] if len(sys.argv) > 1 else "gemm"
torch.manual_seed(0)
K, N = 4096, 4096
M = 2048 if which == "fused" else int(sys.argv[2]) if len(sys.argv) > 2 else 2048
N = 512 if (which == "gemm" and M == 2048) else N
X = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
W = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
X_cal = torch.randn(256, K, dtype=torch.float32, device='cuda')
T64 = compute_wush_transform(X_cal, W.float(), group_size=32)
T = T64.to(torch.bfloat16)
W_w = transform_weights(W.double(), T64).to(torch.bfloat16).contiguous()
wqw, wsw = quant_int4(W_w)
if which == "gemm":
    xqw, xsw = fused_transform_quant_int4_wmma(X, T)
    for _ in range(5): y = w4a4_gemm_s4(xqw, xsw, wqw, wsw)
else:
    for _ in range(5): xqw, xsw = fused_transform_quant_int4_wmma(X, T)
torch.cuda.synchronize()
print("done", which)
