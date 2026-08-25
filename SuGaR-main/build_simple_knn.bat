@echo off
REM Build simple-knn CUDA extension 
REM Must run from D:\CAAS\SuGaR-main

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1
set DISTUTILS_USE_SDK=1
set FORCE_CUDA=1
set TORCH_CUDA_ARCH_LIST=8.6
set MAX_JOBS=4
set PATH=D:\CAAS\SuGaR-main\.venv\Scripts;%PATH%

cd /d D:\CAAS\SuGaR-main\gaussian_splatting\submodules\simple-knn
python setup.py install

echo.
echo EXIT CODE: %ERRORLEVEL%
pause
