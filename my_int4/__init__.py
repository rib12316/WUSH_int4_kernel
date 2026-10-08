# my_int4 包入口: _C.pyd 由 setup.py build_ext --inplace 编译后放本目录
# [P14] 先载入 torch 再 import _C —— Windows 下 torch 会注册其 CUDA DLL
# 搜索路径, 顺序反了会报 "DLL load failed" (P12 实录)。
import torch  # noqa: F401

from my_int4._C import quant_int4  # noqa: F401
