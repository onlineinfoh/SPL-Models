#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

echo "[1/3] Training (train.py)"
# "${PYTHON}" "${ROOT}/train.py"

echo "[2/3] Inference (infer_probs_tight.py)"
"${PYTHON}" "${ROOT}/infer_probs_tight.py"

echo "[3/3] Heatmaps (heatmap.py)"
"${PYTHON}" "${ROOT}/heatmap.py"
