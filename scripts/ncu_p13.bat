@echo off
rem P13 ncu profile driver: arg1 = report name, arg2 = prof_target.py M, arg3 = skip, arg4 = count
set NCU=C:\Program Files\NVIDIA Corporation\Nsight Compute 2023.3.1\ncu.bat
set PY=D:\code\CUDA\open-source-ops\.venv\Scripts\python.exe
cd /d D:\code\CUDA\open-source-ops\my_int4
call "%NCU%" --set full --target-processes all -k "regex:w4a4_gemm_s4" --launch-skip %3 --launch-count %4 -o %1 -f %PY% bench/prof_target.py gemm %2
