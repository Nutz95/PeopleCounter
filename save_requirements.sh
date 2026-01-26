#!/usr/bin/env bash
# Save current venv packages to requirements.txt
VENV_DIR=.venv
PY="$VENV_DIR/Scripts/python.exe"
if [ ! -f "$PY" ]; then
  echo "Virtualenv not found. Create it first: python -m venv .venv"
  exit 1
fi
"$PY" -m pip freeze > requirements.txt
echo "requirements.txt updated." 
