@echo off
setlocal

REM Launch app2.py with the project virtual environment.
set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv1\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo Virtual env python not found at %PYTHON%
    exit /b 1
)

pushd "%ROOT%"
"%PYTHON%" app2.py
popd
