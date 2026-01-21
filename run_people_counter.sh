#!/usr/bin/env bash
set -euo pipefail

# Run from repository root (script dir)
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

# Create venv if missing
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment .venv..."
  python -m venv .venv
fi

# Activate venv (supports POSIX and Git Bash/PowerShell paths)
if [ -f ".venv/bin/activate" ]; then
  # Linux / WSL / macOS
  source ".venv/bin/activate"
  PYTHON=".venv/bin/python"
elif [ -f ".venv/Scripts/activate" ]; then
  # Git Bash on Windows
  source ".venv/Scripts/activate"
  PYTHON=".venv/Scripts/python"
else
  echo "No venv activation script found in .venv"
  exit 1
fi

# Optionally install requirements if INSTALL_REQS=1
if [ "${INSTALL_REQS:-0}" = "1" ]; then
  echo "Upgrading pip and installing requirements..."
  $PYTHON -m pip install --upgrade pip
  if [ -f requirements.txt ]; then
    $PYTHON -m pip install -r requirements.txt
  else
    echo "requirements.txt not found, skipping pip install."
  fi
fi

# Default script to run (can pass alternative as first arg)
SCRIPT="${1:-camera_app_pipeline.py}"
shift || true

echo "Running: $PYTHON $SCRIPT $*"
$PYTHON "$SCRIPT" "$@"
