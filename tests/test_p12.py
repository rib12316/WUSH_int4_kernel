# ============================================================================
# P12: 快速路线对拍与验证 — per-tensor scale 量化 + CUTLASS s4 + (Build B) CUDA Graph
# ============================================================================
# 口径: 此路线"不计精度"只求算子速度 —— 但正确性对拍仍是铁律:
#   ① 量化 kernel 必须与位镜像参考 100% 逐位一致 (P1 以来的一贯纪律)
#   ② CUTLASS GEMM 对"张量级 scale 的精确反量化"的 fp64 仲裁 <5%
#   ③ (Build B) graph 捕获后 replay 与 eager 同链位一致 (FLAG: cudaFuncSetAttribute)
# 镜像纪律 (handoff 踩坑记录):
#   - kernel 用乘 inv_scale(fp32) 不用除法 → 参考也必须乘 0-dim fp32 CUDA tensor
#     (tensor/py_scalar 除法会被优化为乘倒数, 差 1ulp 翻平局码)
#   - __float2int_rn = round-half-to-even = fp32 上的 torch.round
#   - kernel 先 rint 后 clamp[-7,7]; 偶元素低 nibble
import torch, sys, os
import os as _os, sys as _sys  # [P14] 仓库根上路径 (tests/bench/scripts 共用)
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from my_int4 import _C

torch.manual_seed(0)


def ref_pack(x_bf16, scale):
    """位镜像参考: 严格复刻 quant_int4_ts_kernel 的 fp 运算序 → packed uint8 (..., K/2)"""
    inv = torch.tensor(1.0 / scale, dtype=torch.float32, device=x_bf16.device)  # (float)(1.0/scale) = 7/amax
    q = torch.round(x_bf16.float() * inv)          # fp32 真乘法 (0-dim fp32 tensor)
    qi = q.clamp_(-7, 7).to(torch.int8) & 0xF      # rint 后 clamp; & 0xF = 补码 nibble
    return (qi[..., 0::2] | (qi[..., 1::2] << 4)).to(torch.uint8)


def dq_ts(qb, scale):
    """张量级 scale 反量化 (fp32) — 沿 test_s4c 的 dq_t"""
    lo = ((qb & 0xF).to(torch.int16) << 12) >> 12
    hi = ((qb >> 4).to(torch.int16) << 12) >> 12
    d = torch.stack([lo, hi], dim=-1).flatten(-2).float()
    return d * scale


def amax_scale(t):
    """t(bf16) → (python float scale=amax/7, fp32 amax tensor)"""
    amax = t.float().abs().max().clamp_min(1e-8)
    return (amax / 7.0).item(), amax


print("== P12-1 位对拍: quant_int4_tensor_scale vs 位镜像参考 ==")
bit_cases = []
for shape in [(1, 4096), (64, 4096), (2048, 512), (4096, 4096)]:
    bit_cases.append((torch.randn(*shape, dtype=torch.bfloat16, device='cuda'), "randn"))
for mag in (1e-3, 1e3):
    bit_cases.append(((torch.rand(64, 4096, device='cuda') * 2 - 1) * mag, f"uniform±{mag:g}"))
for x1 in (torch.randn(96, dtype=torch.bfloat16, device='cuda'),
           torch.randn(8224, dtype=torch.bfloat16, device='cuda').view(2, -1)):
    bit_cases.append((x1, "尾块"))
for scale_sweep in (1e-3, 0.37, 1.0, 256.0):
    base = torch.linspace(-8.5, 8.5, 4096, device='cuda')
    bit_cases.append(((base * scale_sweep).to(torch.bfloat16), f"clamp扫描s={scale_sweep:g}"))

total_bad = 0
for X, tag in bit_cases:
    if X.dtype != torch.bfloat16:
        X = X.to(torch.bfloat16)
    X = X.contiguous()
    sx, _ = amax_scale(X)
    q_kernel = _C.quant_int4_tensor_scale(X, sx)
    q_ref = ref_pack(X, sx).reshape(-1)
    bad = (q_kernel != q_ref).sum().item()
    total_bad += bad
    assert torch.equal(q_kernel, q_ref), f"位对拍失配 {tag}: {bad}/{q_kernel.numel()}"
    print(f"   [{tag:>14}] numel={X.numel():>9} 位一致 OK")
assert total_bad == 0
print("   → 全部位一致率 1.0")

print("\n== P12-2 信息项: 真除法变体的失配量化 (标量除法陷阱的实证) ==")
X = torch.randn(64, 4096, dtype=torch.bfloat16, device='cuda')
sx, _ = amax_scale(X)
q_kernel = _C.quant_int4_tensor_scale(X, sx)
q_div = torch.round(X.float() / torch.tensor(sx, dtype=torch.float32, device='cuda'))
qi = q_div.clamp_(-7, 7).to(torch.int8) & 0xF
q_div_pack = (qi[..., 0::2] | (qi[..., 1::2] << 4)).to(torch.uint8).reshape(-1)
mism = (q_kernel != q_div_pack)
frac = mism.float().mean().item()
k_lo = dq_ts(q_kernel.view(64, 2048), sx).view(64, 4096)
k_hi = dq_ts(q_div_pack.view(64, 2048), sx).view(64, 4096)
nibble_dist = (k_lo - k_hi).abs().max().item() / sx
print(f"   真除法 vs 乘倒数: 失配字节 {mism.sum().item()}/{mism.numel()} ({frac:.2e}), nibble 距 ≤ {nibble_dist:.0f}")
assert frac < 1e-3 and nibble_dist <= 1.0, "真除法偏离超界 (乘法镜像假设被破坏?)"

print("\n== P12-3 反量化界 (amax 定标时 |x-dq| ≤ 0.51·scale) ==")
for shape in [(1, 4096), (64, 4096), (2048, 512)]:
    X = torch.randn(*shape, dtype=torch.bfloat16, device='cuda').contiguous()
    sx, _ = amax_scale(X)
    d = (dq_ts(_C.quant_int4_tensor_scale(X, sx).view(*shape[:-1], -1), sx) - X.float()).abs().max().item()
    assert d <= 0.51 * sx, (shape, d, sx)
    print(f"   {shape}: max|Δ|={d:.4e} ≤ 0.51·s={0.51 * sx:.4e} OK")

print("\n== P12-4 e2e fp64 仲裁 (量化 → CUTLASS s4, 张量级口径) ==")
for (M, N, K) in [(64, 96, 128), (1, 256, 64), (300, 500, 512), (130, 512, 4096), (8, 4096, 4096)]:
    X = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    W = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    sx, _ = amax_scale(X); sw, _ = amax_scale(W)
    xq = _C.quant_int4_tensor_scale(X, sx); wq = _C.quant_int4_tensor_scale(W, sw).view(N, -1)
    y = _C.w4a4_gemm_cutlass(xq, wq, sx * sw)
    y64 = dq_ts(xq.view(M, -1), sx).double() @ dq_ts(wq.view(N, -1), sw).double().t()
    d = (y.double() - y64).abs().max().item()
    tol = 0.05 * max(1.0, y64.abs().max().item())
    print(f"   {M}x{N}x{K}: 离真值={d:.4f} (|Y|={y64.abs().max().item():.1f})")
    assert d < tol, (M, N, K, d, tol)

# ───────────────────────── Build B (graph 化) 段 ─────────────────────────
if not (hasattr(_C, 'quant_int4_tensor_scale_dev') and hasattr(_C, 'w4a4_gemm_cutlass_dev')):
    print("\n(未检测到 _dev 符号: Build B 未完成, 跳过 dev/graph 段)")
    sys.exit(0)

print("\n== P12-5 dev 版位对拍 (inv_scale 从设备内存读) ==")
for X, tag in [(torch.randn(64, 4096, dtype=torch.bfloat16, device='cuda'), "randn"),
               (torch.randn(8224, dtype=torch.bfloat16, device='cuda'), "尾块1D")]:
    amax = X.float().abs().max().clamp_min(1e-8)
    inv_t = 7.0 / amax                      # GPU 链产物 (fp32 0-dim) — 镜像乘同一个张量
    q_kernel = _C.quant_int4_tensor_scale_dev(X.contiguous(), inv_t)
    q = torch.round(X.float() * inv_t)
    qi = q.clamp_(-7, 7).to(torch.int8) & 0xF
    q_ref = (qi.reshape(-1, 2)[:, 0] | (qi.reshape(-1, 2)[:, 1] << 4)).to(torch.uint8)
    assert torch.equal(q_kernel, q_ref), tag
    print(f"   [{tag}] 位一致 OK")


def fast_step(X, wqt, sw_host):
    """全 GPU 快速路线链 (可捕获): amax→inv→量化→CUTLASS(alpha_ptr)"""
    amax = X.abs().max().float().clamp_(min=1e-8)
    inv = 7.0 / amax
    xq = _C.quant_int4_tensor_scale_dev(X, inv)
    alpha = amax * (sw_host / 7.0)
    return _C.w4a4_gemm_cutlass_dev(xq, wqt, alpha)


def graphed(step):
    for _ in range(3): out = step()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out = step()
    return g, out


print("\n== P12-6 CUDA Graph 捕获 + replay 位一致 (FLAG 哨兵, 不容 try/except) ==")
for (M, N, K) in [(64, 4096, 4096), (2048, 512, 4096), (8, 4096, 4096)]:
    X = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    W = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
    sw, _ = amax_scale(W)
    wqt = _C.quant_int4_tensor_scale(W.contiguous(), sw).view(N, -1)
    y0 = fast_step(X, wqt, sw).clone()      # eager 基准 (同 dev 链, 防 host/GPU alpha 差 1ulp 假阳性)
    g, out = graphed(lambda: fast_step(X, wqt, sw))
    g.replay(); torch.cuda.synchronize()
    assert torch.equal(out, y0), (M, N)
    print(f"   {M}x{N}x{K}: replay == eager (位一致) OK")

print("\n== P12-7 信息项: outlier 数据 e2e 相对误差 (不计精度, 仅记录) ==")
M, N, K, GT = 2048, 512, 4096, 32
X = torch.randn(M, K, dtype=torch.bfloat16, device='cuda'); X[:, 5::GT] *= 15.0
W = torch.randn(N, K, dtype=torch.bfloat16, device='cuda'); W[:, 12::GT] *= 8.0
Yt = X.double() @ W.double().t()
sx, _ = amax_scale(X); sw, _ = amax_scale(W)
xq = _C.quant_int4_tensor_scale(X, sx); wq = _C.quant_int4_tensor_scale(W, sw).view(N, -1)
y = _C.w4a4_gemm_cutlass(xq, wq, sx * sw)
e = ((y.double() - Yt).norm() / Yt.norm()).item()
print(f"   快速路线 outlier e2e 误差: {e:.4%}  (逐组路线基线 7.83%; 本路线不计精度)")

print("\nP12 测试全部通过")
