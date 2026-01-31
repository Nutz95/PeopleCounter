@echo off
SETLOCAL EnableDelayedExpansion

echo ---------------------------------------------------------
echo Initialisation de l'environnement Python pour le Bridge...
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
    echo [+] Creation de l'environnement virtuel (venv_bridge)...
    python -m venv venv_bridge
)

:: Installation des paquets
echo [+] Verification des dependances...
call venv_bridge\Scripts\activate
python -m pip install --upgrade pip
python -m pip install flask opencv-python

echo.
echo [+] Lancement du bridge...
python camera_bridge.py

pause
