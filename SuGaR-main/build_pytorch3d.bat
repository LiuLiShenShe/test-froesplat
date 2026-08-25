@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1
set DISTUTILS_USE_SDK=1
set FORCE_CUDA=1
set TORCH_CUDA_ARCH_LIST=8.6
set MAX_JOBS=4
set PATH=D:\CAAS\SuGaR-main\.venv\Scripts;%PATH%

echo Testing cl.exe...
where cl
echo Testing nvcc...
where nvcc

cd /d D:\CAAS\SuGaR-main\pytorch3d
rd /s /q build 2>nul

echo Starting pytorch3d build...
D:\CAAS\SuGaR-main\.venv\Scripts\python.exe setup.py install
echo EXIT CODE: %ERRORLEVEL%
