# WUSH_int4_kernel

INT4 W4A4 高性能推理算子 —— 按 [WUSH 论文](https://arxiv.org/abs/2512.00956)
(ICML 2026, [官方仓库](https://github.com/IST-DASLab/WUSH))的 INT4 口径在
NVIDIA Turing (sm75, RTX 2060) 上的完整自研实现。

WUSH 官方只发布了 FP4 (MXFP4/NVFP4) 算子；论文中的 INT4（G=32 逐组 σ 截断 + BF16 scale，
MSE 最优截断常数 c\*=2.473650σ 为数值求得的 INT4 专属值）没有实现 —— 本仓库补全这一半。

## 亮点

- **论文 INT4 口径全程保持**：逐组 G=32 σ 截断量化（位级对拍参考实现）、WUSH 变换
  （在线融合）、原生 `mma.m8n8k32.s4` Tensor Core GEMM + k 组边界 scale 累加、
  half2 stage-flush 数值路径（outlier e2e 误差 7.81% vs 裸 INT4 53%）；
- **decode/小 batch 段稳定超越 fp16 cuBLAS**（真实 LLM 层形状，CUDA Graph + med5）：
  - Llama-70B FFN 层 (8192×57344)：M=1 **3.14×**，M=8 1.30×，M=32 1.31×
  - Qwen3-8B FFN 层 (4096×24576)：M=1 **3.07×**，M=8 1.37×，M=32 1.16×
- **per-tensor scale 快速路线**（不计精度，作为 sm75 s4 Tensor Core 性能上限参照）：
  4096³ e2e **3.31×** vs fp16（CUTLASS s4 GEMM 单核 5-6×）；
- **13 轮 ncu 证据链**（`docs/NCU_FINDINGS.md`）：sm75 上逐组 scale 路线的三堵结构墙
  （XU 转换墙 / MIO 融合墙 / 大 M 算术墙）全部实测闭合 —— prefill 段 0.8-0.9× 是
  硬件代差所致，升级 sm80+/sm100 是唯一解锁路径（云端验证套件已备好）。

## 目录结构

```
csrc/           CUDA kernels (12 .cu + bindings.cpp)
  quant_int4.cu        σ 截断量化 (位级对拍)
  gemm_s4.cu           主线: 原生 s4 mma + 逐组 scale + half2 stage-flush + splitk 原子 epilogue
  gemm_cutlass_int4.cu CUTLASS s4 (per-tensor, graph 安全直 launch + alpha_ptr)
  fast_path.cu         per-tensor 快速路线量化 (eager + graph 化双入口)
  gemv_int4.cu         decode 族 (M=1 / M≤8)
  fused_tq_*.cu        WUSH 变换+量化融合 (WMMA / 纯寄存器 shuffle / smallm)
  gemm_int4/wmma*.cu   前期版本 (对拍演进史)
my_int4/        Python 包: _C 扩展 + rtn_int4 (参考实现) + wush_transform (变换管线)
tests/          12 件套回归 (位对拍 / fp64 仲裁 / outlier 精度硬闸 / graph 位一致)
bench/          基准 (P12 快速路线 / P13 9形态矩阵 / P13 官方 LLM 形态) + ncu 驱动
scripts/        build.bat / run_sm100.py (云端一键验证) / ncu 采集脚本
docs/           SPEC.md (语义规格) / NCU_FINDINGS.md (13轮证据) / SM100_VERIFY.md
```

## 环境与构建

- Windows + python.org Python 3.11 + PyTorch 2.8 (cu126) + CUDA 12.3 + VS2022 Build Tools
- **外部依赖**：WUSH 仓库签出（提供 `block_transform`/`compute_gram` 数学模块与 vendored
  CUTLASS 头）。默认取兄弟目录 `../WUSH`，可用环境变量 `WUSH_ROOT` / `CUTLASS_INC` 覆盖：

```bash
git clone --recurse-submodules https://github.com/IST-DASLab/WUSH.git   # 放在本仓库旁边
scripts\build.bat        # vcvars + DISTUTILS_USE_SDK + setup.py build_ext --inplace
```

## 测试与基准

```bash
python tests\test_p13.py           # 主线 GEMM 闸门 (fp64 仲裁 + outlier 硬闸 + graph)
for /f %t in ('dir /b tests\test_*.py') do python tests\%t   # 全回归
python bench\exp_p13_official.py  # 真实 LLM 层形态基准 (Qwen3-8B / Llama-70B)
python bench\exp_p13_matrix.py    # 9 形态常备矩阵
python bench\exp_p12_fastpath.py  # per-tensor 快速路线
```

## 结果速览（RTX 2060，CUDA Graph med5，vs fp16 cuBLAS）

| 路线 | decode (M=1-32) | prefill (M≥128) |
|---|---|---|
| 论文 INT4 口径（逐组 σ） | **1.0-3.14×** | 0.64-0.96×（结构墙，见 docs） |
| per-tensor 快速（不计精度） | 1.5-2.3× | 4096³ **3.31×** |

精度（outlier 数据 e2e 相对误差）：逐组路线 **7.81%**（裸 INT4 53%）；
per-tensor 路线 99.7%（仅适用于已标定/干净动态域，作性能上限参照）。

## 致谢

- WUSH: Chen, Egiazarian, Castro, Hoefler, Alistarh. *Near-Optimal Adaptive Transforms
  for LLM Quantization.* ICML 2026. （算法与上游 FP4 算子参照）
- CUTLASS (NVIDIA) —— s4 Tensor Core 原子与 GEMM 骨架。
