@echo off
SETLOCAL EnableDelayedExpansion

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo ---------------------------------------------------------
echo Initialisation du Bridge Python
echo ---------------------------------------------------------

:: Vérifier si Python est installé
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python n'est pas installe ou pas dans le PATH.
    echo Veuillez installer Python depuis python.org.
    pause
    exit /b
)

:: Création du venv s'il n'existe pas
if not exist "venv_bridge" (
    echo [+] Creation de l'environnement virtuel venv_bridge...
    python -m venv venv_bridge
)

:: Installation des paquets
echo [+] Verification des dependances...
venv_bridge\Scripts\python.exe -m pip install --upgrade pip
venv_bridge\Scripts\python.exe -m pip install -r "%SCRIPT_DIR%requirements-media-bridge.txt"

echo.
echo [+] Lancement du media bridge...
:: Recuperer le chemin complet du fichier video initial
::for %%I in ("%~dp0\ref_videos\example.mp4") do set FULL_PATH=%%~fI
::for %%I in ("%~dp0\ref_videos\Tokyo.mp4") do set FULL_PATH=%%~fI
::for %%I in ("%~dp0\ref_videos\Crowd_Shotcut.mp4") do set FULL_PATH=%%~fI
for %%I in ("%SCRIPT_DIR%ref_videos\Crowd.mp4") do set FULL_PATH=%%~fI

::for %%I in ("%~dp0\ref_videos\People_walking.mp4") do set FULL_PATH=%%~fI
set MEDIA_DIR=E:\SequencesVideo\Crowd
if not exist "%MEDIA_DIR%" set MEDIA_DIR=%SCRIPT_DIR%ref_videos
set IMAGE_DIR=%SCRIPT_DIR%ref_images
set FULL_PATH=
for %%I in ("%MEDIA_DIR%\*.mp4") do if not defined FULL_PATH set FULL_PATH=%%~fI
if not defined FULL_PATH for %%I in ("%SCRIPT_DIR%ref_videos\Crowd.mp4") do set FULL_PATH=%%~fI

venv_bridge\Scripts\python.exe -m media_bridge --media-dir "%MEDIA_DIR%" --image-dir "%IMAGE_DIR%" --input-file "%FULL_PATH%" --resolution 4K --fps 30 --bitrate 50000 --encoder h264_qsv

pause
