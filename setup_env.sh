#!/usr/bin/env bash
# ------------------------------------------------------------------
# Bootstrap script: Python venv, requirements, dataset download.
# Usage: ./setup_env.sh
# ------------------------------------------------------------------
set -euo pipefail

PY_VER="3.11"
VENV_DIR=".venv"

echo "🔧  Creating Python ${PY_VER} virtual environment..."
python${PY_VER} -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "⬇️  Installing Python requirements..."
pip install --upgrade pip
pip install -r requirements.txt kaggle

echo "📦  Pulling Online Payments Fraud Detection dataset..."
DATA_DIR="data/raw"
mkdir -p "${DATA_DIR}"
kaggle datasets download -d rupakroy/online-payments-fraud-detection-dataset -p "${DATA_DIR}"
unzip -o "${DATA_DIR}"/online-payments-fraud-detection-dataset.zip -d "${DATA_DIR}"

echo "✅  Setup complete. Virtual env is active—run ./run_local.sh next."
