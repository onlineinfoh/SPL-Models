#!/usr/bin/env bash
# Original checkpoint pipeline; training stays disabled in this driver.
# Inference and Grad-CAM use 300 px and write to separate corrected directories.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

echo "[1/3] Using existing checkpoints (training disabled)"
# "${PYTHON}" "${ROOT}/train.py"

echo "[2/3] Inference (infer_probs_tight.py)"
"${PYTHON}" "${ROOT}/infer_probs_tight.py"

echo "[3/3] Heatmaps (heatmap.py)"
"${PYTHON}" "${ROOT}/heatmap.py"
