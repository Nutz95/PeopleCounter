@echo off
SETLOCAL EnableDelayedExpansion

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

:: camera_bridge.py n'utilise que la bibliotheque standard Python.
:: On evite tout pip install ici pour que la demo puisse tourner hors-ligne.
echo [+] Activation de l'environnement virtuel...
call venv_bridge\Scripts\activate
echo [+] Aucune dependance Python externe requise.

echo.
echo [+] Lancement du bridge...
venv_bridge\Scripts\python.exe camera_bridge.py

pause
