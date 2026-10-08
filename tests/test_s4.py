# S4 对拍 + 性能: WMMA v1 vs v0 vs fp16
import torch, sys, os
import os as _os, sys as _sys  # [P14] 仓库根上路径 (tests/bench/scripts 共用)
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from my_int4.rtn_int4 import quant_int4, dequant_int4, w4a4_matmul_reference
from my_int4._C import quant_int4, w4a4_gemm, w4a4_gemm_v1

def check(X, W, name):
    xq, xs = quant_int4(X)
    wq, ws = quant_int4(W)
    y_v0 = w4a4_gemm(xq, xs, wq, ws)
    y_v1 = w4a4_gemm_v1(xq, xs, wq, ws)
    y_ref = w4a4_matmul_reference(xq, xs, wq, ws)
    xd = dequant_int4(xq, xs).double(); wd = dequant_int4(wq, ws).double()
    y64 = xd @ wd.t()
    scale = max(1.0, y64.abs().max().item())
    d_v0 = (y_v0.double() - y64).abs().max().item()
    d_v1 = (y_v1.double() - y64).abs().max().item()
    d_01 = (y_v0.float() - y_v1.float()).abs().max().item()
    print(f"   {name}: v0离真值={d_v0:.4f}  v1离真值={d_v1:.4f}  v0/v1互差={d_01:.4f}  (|Y|={scale:.1f})")
    assert d_v1 < 0.05 * scale, (name, "v1 偏离 fp64 真值")
    assert d_01 < 0.05 * scale, (name, "v0/v1 互相偏离")

torch.manual_seed(0)
print("== S4 正确性 ==")
check(torch.randn(64, 128, dtype=torch.bfloat16, device='cuda'),
      torch.randn(96, 128, dtype=torch.bfloat16, device='cuda'), "64x96x128")
check(torch.randn(1, 128, dtype=torch.bfloat16, device='cuda'),
      torch.randn(64, 128, dtype=torch.bfloat16, device='cuda'), "M=1")
check(torch.randn(257, 96, dtype=torch.bfloat16, device='cuda'),
      torch.randn(33, 96, dtype=torch.bfloat16, device='cuda'), "非整块(补零路径)")
check(torch.randn(130, 4096, dtype=torch.bfloat16, device='cuda'),
      torch.randn(512, 4096, dtype=torch.bfloat16, device='cuda'), "多tile 130x512x4096")

print("\n== S4 性能 (ms, 越小越好) ==")
def cuda_time(fn, reps=30, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / reps

print(f"{'M':>5} {'N':>6} {'K':>6} | {'v0':>9} {'v1':>9} {'fp16':>9} | {'v1/v0':>6} {'v1/fp16':>8}")
print("-" * 70)
for (M, N, K) in [(1, 4096, 4096), (64, 4096, 4096), (1024, 4096, 4096), (4096, 4096, 4096)]:
    X = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    W = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    xq, xs = quant_int4(X); wq, ws = quant_int4(W)
    X16, W16 = X.half(), W.half()
    reps = 30 if M <= 64 else 5
    t_v0 = cuda_time(lambda: w4a4_gemm(xq, xs, wq, ws), reps=reps, warmup=3)
    t_v1 = cuda_time(lambda: w4a4_gemm_v1(xq, xs, wq, ws), reps=reps, warmup=3)
    t_fp16 = cuda_time(lambda: X16 @ W16.t(), reps=reps, warmup=3)
    print(f"{M:>5} {N:>6} {K:>6} | {t_v0:>9.3f} {t_v1:>9.3f} {t_fp16:>9.3f} | {t_v0/t_v1:>5.1f}x {t_fp16/t_v1:>7.2f}x")
