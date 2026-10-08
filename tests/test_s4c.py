# S4c 对拍 + 性能: CUTLASS INT4 Tensor Core (张量级 scale) vs 手写各版 vs fp16
# 口径: CUTLASS 版是"每张量一组"的量化 —— 与分组版是不同的量化, 必须各对
# 各的 fp64 真值负责, 不能互相逐位比。
import torch, sys, os
import os as _os, sys as _sys  # [P14] 仓库根上路径 (tests/bench/scripts 共用)
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from my_int4._C import quant_int4, w4a4_gemm, w4a4_gemm_v1, w4a4_gemm_cutlass

def quant_tensor(x):
    """张量级对称 INT4: 全张量一个 scale (S4c 的量化口径)"""
    amax = x.float().abs().max().clamp_min(1e-8)
    scale = (amax / 7.0).item()
    q = torch.clamp(torch.round(x.float() / scale), -8, 7).to(torch.int8)
    byte = ((q[..., 0::2] & 0xF) | (q[..., 1::2] & 0xF) << 4).to(torch.uint8)
    return byte, scale

def check(X, W, name):
    # CUTLASS 版 (张量级 scale)
    xq, s_x = quant_tensor(X)
    wq, s_w = quant_tensor(W)
    y_cl = w4a4_gemm_cutlass(xq, wq, s_x * s_w)
    # fp64 真值 (张量级口径的精确反量化)
    def dq_t(qb, s):
        lo = ((qb & 0xF).to(torch.int16) << 12) >> 12
        hi = ((qb >> 4).to(torch.int16) << 12) >> 12
        d = torch.stack([lo, hi], dim=-1).flatten(-2).double() * s
        return d
    y64 = dq_t(xq, s_x) @ dq_t(wq, s_w).t()
    d_cl = (y_cl.double() - y64).abs().max().item()
    scale = max(1.0, y64.abs().max().item())
    print(f"   {name}: CUTLASS离真值={d_cl:.4f} (|Y|={scale:.1f})")
    assert d_cl < 0.05 * scale, (name, "CUTLASS 偏离 fp64 真值")

torch.manual_seed(0)
print("== S4c 正确性 ==")
check(torch.randn(64, 128, dtype=torch.bfloat16, device='cuda'),
      torch.randn(96, 128, dtype=torch.bfloat16, device='cuda'), "64x96x128")
check(torch.randn(1, 256, dtype=torch.bfloat16, device='cuda'),
      torch.randn(64, 256, dtype=torch.bfloat16, device='cuda'), "M=1")
check(torch.randn(300, 512, dtype=torch.bfloat16, device='cuda'),
      torch.randn(500, 512, dtype=torch.bfloat16, device='cuda'), "非整块 300x500x512")
check(torch.randn(130, 4096, dtype=torch.bfloat16, device='cuda'),
      torch.randn(512, 4096, dtype=torch.bfloat16, device='cuda'), "多tile 130x512x4096")

print("\n== S4c 性能 (ms) ==")
def cuda_time(fn, reps=30, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / reps

print(f"{'M':>5} {'N':>6} {'K':>6} | {'v0':>9} {'v1':>9} {'CLAS':>9} {'fp16':>9} | {'CLAS/v1':>7} {'CLAS/fp16':>9}")
print("-" * 82)
for (M, N, K) in [(1, 4096, 4096), (64, 4096, 4096), (1024, 4096, 4096), (4096, 4096, 4096)]:
    X = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    W = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    xq, xs = quant_int4(X); wq, ws = quant_int4(W)
    xqt, s_xt = quant_tensor(X); wqt, s_wt = quant_tensor(W)
    X16, W16 = X.half(), W.half()
    reps = 30 if M <= 64 else 5
    t_v0 = cuda_time(lambda: w4a4_gemm(xq, xs, wq, ws), reps=reps, warmup=3)
    t_v1 = cuda_time(lambda: w4a4_gemm_v1(xq, xs, wq, ws), reps=reps, warmup=3)
    t_cl = cuda_time(lambda: w4a4_gemm_cutlass(xqt, wqt, s_xt * s_wt), reps=reps, warmup=3)
    t_fp16 = cuda_time(lambda: X16 @ W16.t(), reps=reps, warmup=3)
    print(f"{M:>5} {N:>6} {K:>6} | {t_v0:>9.3f} {t_v1:>9.3f} {t_cl:>9.3f} {t_fp16:>9.3f} | {t_v1/t_cl:>6.2f}x {t_fp16/t_cl:>8.2f}x")
