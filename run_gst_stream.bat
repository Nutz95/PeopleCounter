@echo off
SETLOCAL EnableDelayedExpansion

set "ROOT_DIR=%~dp0"
set "MEDIA_DIR_SETTING=windows\ref_videos"
set "IMAGE_DIR_SETTING=windows\ref_images"

set "WIN_DIR=%~dp0windows"
cd /d "%WIN_DIR%"
set "REQ_FILE=%WIN_DIR%\requirements-gst-bridge.txt"
set "VENV_DIR=venv_bridge"
set "PY_VERSION_FILE=%TEMP%\peoplecounter_gst_bridge_python_version.txt"

call :resolve_user_path "%MEDIA_DIR_SETTING%" MEDIA_DIR_RESOLVED
call :resolve_user_path "%IMAGE_DIR_SETTING%" IMAGE_DIR_RESOLVED

for %%V in (GST_PLUGIN_PATH GST_PLUGIN_PATH_1_0 GST_PLUGIN_SYSTEM_PATH GST_PLUGIN_SYSTEM_PATH_1_0 GST_REGISTRY GST_REGISTRY_FORK GST_REGISTRY_REUSE_PLUGIN_SCANNER GST_PLUGIN_SCANNER GST_PLUGIN_SCANNER_1_0) do (
    set "%%V="
)

if "%GST_BRIDGE_PYTHON_VERSION%"=="" set "GST_BRIDGE_PYTHON_VERSION=3.12"

echo.
echo  PeopleCounter GStreamer + MediaMTX bridge
echo  ==========================================
echo.
echo [+] Target Python version: %GST_BRIDGE_PYTHON_VERSION%

py -%GST_BRIDGE_PYTHON_VERSION% -c "import sys" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python %GST_BRIDGE_PYTHON_VERSION% is not installed.
    echo         Install it first or set GST_BRIDGE_PYTHON_VERSION to an installed version.
    echo.
    py --list
    call :maybe_pause
    exit /b 1
)

set "REBUILD_VENV="
if exist "%VENV_DIR%\Scripts\python.exe" (
    del /q "%PY_VERSION_FILE%" >nul 2>&1
    "%VENV_DIR%\Scripts\python.exe" -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" > "%PY_VERSION_FILE%" 2>nul
    set "CURRENT_VENV_PY="
    if exist "%PY_VERSION_FILE%" set /p CURRENT_VENV_PY=<"%PY_VERSION_FILE%"
    del /q "%PY_VERSION_FILE%" >nul 2>&1
    if not "!CURRENT_VENV_PY!"=="%GST_BRIDGE_PYTHON_VERSION%" (
        echo [!] Existing venv uses Python !CURRENT_VENV_PY!; rebuilding for Python %GST_BRIDGE_PYTHON_VERSION% ...
        rmdir /s /q "%VENV_DIR%"
        if exist "%VENV_DIR%" (
            echo [ERROR] Could not remove existing venv at %CD%\%VENV_DIR%.
            call :maybe_pause
            exit /b 1
        )
    )
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [+] Creating Python virtual environment ...
    py -%GST_BRIDGE_PYTHON_VERSION% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Could not create venv with Python %GST_BRIDGE_PYTHON_VERSION%.
        call :maybe_pause
        exit /b 1
    )
)

echo [+] Checking Python dependencies ...
"%VENV_DIR%\Scripts\python.exe" -m pip install --quiet --upgrade pip
if errorlevel 1 (
    echo [ERROR] pip upgrade failed.
    call :maybe_pause
    exit /b 1
)
"%VENV_DIR%\Scripts\python.exe" -m pip install --quiet -r "%REQ_FILE%"
if errorlevel 1 (
    echo [ERROR] Python dependency installation failed.
    call :maybe_pause
    exit /b 1
)

echo [+] Launching gst_bridge ...
echo     media dir : %MEDIA_DIR_RESOLVED%
echo     image dir : %IMAGE_DIR_RESOLVED%
echo.
"%VENV_DIR%\Scripts\python.exe" -m gst_bridge.main --media-dir "%MEDIA_DIR_RESOLVED%" --image-dir "%IMAGE_DIR_RESOLVED%" %*

call :maybe_pause
exit /b %ERRORLEVEL%

:resolve_user_path
set "INPUT_PATH=%~1"
if "%INPUT_PATH:~1,1%"==":" (
    set "ABS_PATH=%INPUT_PATH%"
) else if "%INPUT_PATH:~0,2%"=="\\" (
    set "ABS_PATH=%INPUT_PATH%"
) else (
    set "ABS_PATH=%ROOT_DIR%%INPUT_PATH%"
)
for %%I in ("%ABS_PATH%") do set "%~2=%%~fI"
exit /b 0

:maybe_pause
if "%GST_BRIDGE_NO_PAUSE%"=="1" exit /b 0
pause
exit /b 0
