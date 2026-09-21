#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NNUNET_ENV_PYTHON="${REPO_ROOT}/nnunet-env/bin/python"
NNUNET_SOURCE_ROOT="${REPO_ROOT}/nnUNet"

export nnUNet_raw="${REPO_ROOT}/seg-model-training/nnunet/nnUNet_raw"
export nnUNet_preprocessed="${REPO_ROOT}/seg-model-training/nnunet/nnUNet_preprocessed"
export nnUNet_results="${REPO_ROOT}/seg-model-training/nnunet/nnUNet_results"
export PYTHONPATH="${NNUNET_SOURCE_ROOT}:${PYTHONPATH:-}"
export nnUNet_compile="false"
export LD_LIBRARY_PATH="${REPO_ROOT}/nnunet-env/lib/python3.12/site-packages/nvidia/cudnn/lib:${REPO_ROOT}/nnunet-env/lib/python3.12/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}"

INPUT_DIR="${SCRIPT_DIR}/tmp_in"
OUTPUT_DIR="${SCRIPT_DIR}/tmp_out"
MODEL_DIR="${REPO_ROOT}/seg-model-training/nnunet/nnUNet_results/Dataset000_lung/nnUNetTrainer__nnUNetPlans__2d"
FOLD="all"
CHECKPOINT="checkpoint_best.pth"

if [[ -n "${DEVICE:-}" ]]; then
  SELECTED_DEVICE="${DEVICE}"
else
  if "${NNUNET_ENV_PYTHON}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    SELECTED_DEVICE="cuda"
  else
    SELECTED_DEVICE="cpu"
  fi
fi

mkdir -p "${OUTPUT_DIR}"

"${NNUNET_ENV_PYTHON}" -c \
"from nnunetv2.inference.predict_from_raw_data import predict_entry_point_modelfolder; predict_entry_point_modelfolder()" \
  -i "${INPUT_DIR}" \
  -o "${OUTPUT_DIR}" \
  -m "${MODEL_DIR}" \
  -f "${FOLD}" \
  -chk "${CHECKPOINT}" \
  -device "${SELECTED_DEVICE}"
