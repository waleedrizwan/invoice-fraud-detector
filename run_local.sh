#!/usr/bin/env bash
# ------------------------------------------------------------------
# Local dev loop: ETL -> train -> serve FastAPI with reload.
# Usage: ./run_local.sh
# ------------------------------------------------------------------
set -euo pipefail

VENV_DIR=".venv"
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "❌  Virtual env not found. Run ./setup_env.sh first."
  exit 1
fi
source "${VENV_DIR}/bin/activate"

echo "🚰  ETL – cleaning & feature-engineering..."
python src/data/etl.py

echo "🧠  Training model..."
python src/models/train.py

echo "🚀  Launching FastAPI dev server @ http://127.0.0.1:8000 ..."
uvicorn src.api.app:app --reload
