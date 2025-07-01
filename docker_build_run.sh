#!/usr/bin/env bash
# ------------------------------------------------------------------
# Container build & run helper.
# Usage: ./docker_build_run.sh [tag]
# Default tag = latest
# ------------------------------------------------------------------
set -euo pipefail

TAG=${1:-latest}
IMAGE="invoice-fraud-demo:${TAG}"

echo "🐳  Building Docker image '${IMAGE}' ..."
docker build -t "${IMAGE}" .

echo "🏃  Running container on http://localhost:8080 ..."
docker run --rm -p 8080:8080 "${IMAGE}"
