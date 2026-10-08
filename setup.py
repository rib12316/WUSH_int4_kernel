# my_int4 构建脚本: 第2课解剖过的 WUSH setup.py 的最小骨架版
# 纪律: 不加 --use_fast_math (除法必须 IEEE, 否则与参考对拍翻车); /utf-8 保中文注释
import os
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup

# vendored CUTLASS (默认复用兄弟目录 WUSH 的 submodule; 可用环境变量 CUTLASS_INC 覆盖)
_HERE = os.path.dirname(os.path.abspath(__file__))
_CUTLASS_INC = os.environ.get(
    "CUTLASS_INC",
    os.path.normpath(os.path.join(_HERE, "..", "WUSH", "cuda_kernels", "third_party", "cutlass", "include")))

major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
cc = major * 10 + minor
print(f"[my_int4] building for sm_{cc} ({torch.cuda.get_device_name(0)})")

setup(
    name="my_int4",
    ext_modules=[
        CUDAExtension(
            name="my_int4._C",
            sources=[
                "csrc/quant_int4.cu",
                "csrc/gemm_int4.cu",
                "csrc/gemm_wmma.cu",
                "csrc/gemm_wmma2.cu",
                "csrc/gemm_cutlass_int4.cu",
                "csrc/fused_tq_int4.cu",
                "csrc/fused_tq_wmma.cu",
                "csrc/fused_tq_v3.cu",
                "csrc/gemv_int4.cu",
                "csrc/fast_path.cu",
                "csrc/fused_tq_smallm.cu",
                "csrc/gemm_s4.cu",
                "csrc/bindings.cpp",
            ],
            include_dirs=[_CUTLASS_INC],
            define_macros=[("CUTLASS_ENABLE_TENSOR_CORE_MMA", "1")],
            extra_compile_args={
                "cxx": ["/std:c++17", "/utf-8"],
                "nvcc": [
                    "-gencode", f"arch=compute_{cc},code=sm_{cc}",
                    "-std=c++17", "-O3",
                    "--expt-relaxed-constexpr",      # CUTLASS 头文件需要
                    "-Xcompiler", "/utf-8",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
