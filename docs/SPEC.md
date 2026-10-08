# my_int4: 纯 CUDA 的 W4A4 INT4 矩阵乘 (RTX 2060 / sm_75)

参照系: D:\code\CUDA\open-source-ops\WUSH (只读参考, 不改它)
铁律: 参照其"思路与结构", 不逐行抄 —— 我们的格式(INT4/fp16 scale)与它(FP4/E8M0)本质不同

## 语义规格 (第0站已定死)
- 对称量化, 无 zero-point, 分组 G=32
  scale_g = absmax(|x_g|)/7   (fp32 计算, fp16 存储)
  q = clamp(rint(x/scale), -8, 7)     rint = round-half-to-even
- 打包: 2 nibble/字节, 偶元素低4位, int4 补码低4位 (与 WUSH FP4 布局同构, 对拍直觉复用)
- 反量化: nibble 算术移位符号扩展 → ×scale (fp32)
- GEMM: 反量化后 fp16 乘 / fp32 累加 / fp16 输出

## 站点 (每站交付: 完整代码 + 逐行逻辑讲解 + 验收闸门)
- [x] S0 语义规格 (本文件)
- [ ] S1 PyTorch 参考实现 rtn_int4.py + 对拍测试   ← 一切 CUDA 的真值来源
- [ ] S2 CUDA 量化 kernel (csrc/quant_int4.cu + bindings + setup.py)
- [ ] S3 CUDA W4A4 GEMM v0: 反量化+fp16 FMA (正确性优先, 不用 Tensor Core)
- [ ] S4 CUDA GEMM v1: mma.sync fp16 Tensor Core (Turing 原生 m16n8k16)
- [ ] S5 端到端 W4A4 实验 (outlier 数据, 对比 fp16 基线; 复现 WUSH 动机)
- [ ] S6 进阶选做: 激活量化融合 / Turing 原生 INT4 mma (m8n8k32.s4) / INT8 累加

## 与 WUSH 的结构对标 (S2-S4 写作时开卷)
| 我们的部件 | WUSH 参照物 | 学什么 |
|---|---|---|
| 量化 kernel | qutlass epilogue_quant.h AbsMax 分支 | absmax→scale→编码→打包 的流水顺序 |
| 编码函数 | rtn_e2m1 的 even 语义 | 舍入语义与硬件(此处=mma)对齐的思路 |
| GEMM 骨架 | gemm.cu / mx_gemm.h 的 tile/warp 组织 | tile 划分、K 循环、scale 索引 |
| 测试 | tests/test_mxfp4.py | 逐位/容差两级断言的组织 |

## 纪律 (前 6 课真实翻车记录)
对拍先对形状 / 浮点运算顺序逐操作一致 / __float2int_rn 不是 (int)(x+0.5) /
fast_math 最后再开 / 中文注释 /utf-8 / scale 索引沿 K÷32 / int4 必须 sign-extend
