@echo off
rem CPU-only Release build of ik_llama.cpp with MSVC 2022 + Ninja, 2 parallel jobs.
rem Usage: build_ik.bat <source dir> <build dir>
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set SRC=%~1
set BLD=%~2
"C:\Users\esoku\.local\bin\cmake.exe" -S "%SRC%" -B "%BLD%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DGGML_CCACHE=OFF -DGGML_IQK_FA_ALL_QUANTS=OFF -DCMAKE_MAKE_PROGRAM="C:\Users\esoku\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\ninja.exe" || exit /b 1
"C:\Users\esoku\.local\bin\cmake.exe" --build "%BLD%" -j 2 --target llama-bench llama-perplexity llama-cli || exit /b 1
echo BUILD_OK
