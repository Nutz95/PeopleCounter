@echo off
SETLOCAL EnableDelayedExpansion

:: Always run from the windows\ subdirectory regardless of where this bat is called from
set "WIN_DIR=%~dp0windows"
cd /d "%WIN_DIR%"

echo.
echo  Stream Bridge — hardware-accelerated RTSP streaming
echo  =====================================================
echo.

:: Create venv if missing
if not exist "venv_bridge\Scripts\python.exe" (
    echo [+] Creating Python virtual environment ...
    python -m venv venv_bridge
    if errorlevel 1 (
        echo [ERROR] Could not create venv. Make sure Python 3.9+ is in PATH.
        pause
        exit /b 1
    )
)

:: Install / upgrade dependencies (silent unless first run)
echo [+] Checking dependencies ...
venv_bridge\Scripts\python.exe -m pip install --quiet --upgrade pip
venv_bridge\Scripts\python.exe -m pip install --quiet pillow
venv_bridge\Scripts\python.exe -m pip install --quiet matplotlib

echo [+] Launching stream_bridge.py ...
echo.
venv_bridge\Scripts\python.exe stream_bridge.py %*

pause
