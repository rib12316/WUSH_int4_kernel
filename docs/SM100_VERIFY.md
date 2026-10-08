# my_int4 → sm_100 (RTX 5090 / B200) 云端验证套件

本机 (RTX 2060 / sm75 / WDDM) 终态已由三路 ncu 实验闭合证明结构墙:
跨 lane 数据搬运共用 MIO/L1 单管线 (`NCU_FINDINGS.md` P6-P8)。
本套件在 sm_100 实例上一键复现验证 —— 硬件 blockscaled MMA 在 MMA 内部
承载逐组 scale, WUSH 官方口径 5.8× 可达。

## 0. 租实例 (用户操作)
- AutoDL / RunPod / Vast.ai; 镜像选 CUDA 12.8+ (需支持 sm_100a/sm_120a)
- RTX 5090 (sm_120, 32GB) 或 B200 (sm_100)

## 1. 传工程并装环境
```bash
# 本地打包 (Windows 端):
#   tar czf my_int4.tgz -C D:/code/CUDA/open-source-ops my_int4 WUSH/cuda_kernels/third_party
scp my_int4.tgz user@server:~/
ssh user@server
tar xzf my_int4.tgz
cd my_int4
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install scipy statistics
```

## 2. 编译 (setup.py 已自动检测架构 → sm_100/120)
```bash
python setup.py build_ext --inplace
```
注: mma_sm75.h 的 m8n8k32 s4 原子在 sm_80+ 不可用 —— 若报错,
用 `python scripts/run_sm100.py --wmma` 走 WMMA 路线 (nvcuda::wmma 前向兼容),
或从 WUSH 的 cutlass patch 引入 blockscaled 路线 (`gemm.cu` 的 FpGemm)。

## 3. 验证 (正确性 + 全形态性能)
```bash
python scripts/run_sm100.py            # 或 --wmma
```
预期: 8 形态 e2e 全部 >1×; GEMM 单核对标 fp16 2-5×;
       融合 kernel 走 cp.async/TMA 路线后逼近带宽地板。
若 sm_100 上仍未达标 → 运行 `ncu --set full -k regex:fused_tq` 存证。

## 4. 对照基线 (WUSH 官方 5.8× 口径)
```bash
cd ../WUSH/cuda_kernels && uv pip install --no-build-isolation .
python tests/bench_mxfp4_sm100.py   # 需 --format=wush 路径
```
