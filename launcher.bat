@echo off
SETLOCAL EnableDelayedExpansion

:menu
cls
echo ==========================================
echo    PEOPLE COUNTER - HARDWARE SELECTOR
echo ==========================================
echo 1) EXTREME   (NVIDIA RTX 5060 Ti only)
echo 2) HYBRID    (Intel NPU for YOLO + iGPU for Density)
echo 3) BALANCED  (NVIDIA for Density + iGPU for YOLO)
echo 4) FALLBACK  (CPU only)
echo 5) EXIT
echo ==========================================
set /p choice="Select your configuration [1-5]: "

if "%choice%"=="1" set CFG=rtx_extreme
if "%choice%"=="2" set CFG=intel_hybrid
if "%choice%"=="3" set CFG=balanced_tri_chip
if "%choice%"=="4" set CFG=cpu_fallback
if "%choice%"=="5" exit /b

if not defined CFG goto menu

echo Loading %CFG% configuration...

:: Parse .env file manually for Batch
for /f "tokens=1,2 delims==" %%a in (scripts/configs/%CFG%.env) do (
    set "line=%%a"
    if not "!line:~0,1!"=="#" (
        set "%%a=%%b"
    )
)

echo.
echo Running People Counter with:
echo YOLO_DEVICE=%YOLO_DEVICE%
echo DENSITY_DEVICE=%DENSITY_DEVICE%
echo.

python camera_app_pipeline.py
pause
goto menu