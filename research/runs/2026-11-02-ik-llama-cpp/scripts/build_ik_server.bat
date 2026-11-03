@echo off
rem Adds llama-server to the existing CPU-only ik_llama.cpp build (2 parallel jobs).
rem Usage: build_ik_server.bat <build dir>
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
"C:\Users\esoku\.local\bin\cmake.exe" --build "%~1" -j 2 --target llama-server || exit /b 1
echo BUILD_OK
