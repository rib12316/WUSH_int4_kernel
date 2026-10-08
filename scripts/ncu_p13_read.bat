@echo off
rem read an ncu report: arg1 = report file (no extension)
set NCU=C:\Program Files\NVIDIA Corporation\Nsight Compute 2023.3.1\ncu.bat
cd /d D:\code\CUDA\open-source-ops\my_int4
call "%NCU%" --import %1.ncu-rep --page details
