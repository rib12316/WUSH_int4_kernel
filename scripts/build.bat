@echo off
call D:\VS2022BuildTools\VC\Auxiliary\Build\vcvars64.bat >nul 2>&1
set DISTUTILS_USE_SDK=1
set PATH=D:\code\CUDA\open-source-ops\.venv\Scripts;%PATH%
cd /d D:\code\CUDA\open-source-ops\my_int4
D:\code\CUDA\open-source-ops\.venv\Scripts\python.exe setup.py build_ext --inplace
