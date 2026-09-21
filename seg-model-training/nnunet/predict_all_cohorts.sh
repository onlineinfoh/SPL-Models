#!/usr/bin/env bash
# Predict lesion masks for all four cohorts with the retrained nnU-Net.
#
# Uses Dataset001_lungval fold 0 checkpoint_best.pth, which is the first
# nnU-Net checkpoint in this project selected on a genuinely held-out cohort
# (see protocol/DEVIATIONS.md D2).
#
# Writes to predicted_masks_v2/. The published *_model directories under data/
# are NOT touched, so the superseded automatic-mask results stay reproducible.
#
# Run detached:
#   setsid nohup bash seg-model-training/nnunet/predict_all_cohorts.sh \
#     > seg-model-training/nnunet/predict.log 2>&1 < /dev/null &
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

export nnUNet_raw="$REPO/seg-model-training/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$REPO/seg-model-training/nnunet/nnUNet_preprocessed"
export nnUNet_results="$REPO/seg-model-training/nnunet/nnUNet_results"
BIN="$HOME/venvs/prism/bin"
OUT="$REPO/seg-model-training/nnunet/predicted_masks_v2"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "$(ts)  start; checkpoint_best.pth, Dataset001_lungval fold 0"
mkdir -p "$OUT"

fail=0
for spec in "train:data/train/imagesTr" \
            "internal_val:data/val/img_v" \
            "external_test1:data/test1/img_test1" \
            "external_test2:data/test2/img_test2"; do
  name="${spec%%:*}"; src="$REPO/${spec##*:}"; dst="$OUT/$name"
  n_in=$(ls "$src"/*.nii.gz 2>/dev/null | wc -l)
  mkdir -p "$dst"
  echo "$(ts)  predicting $name  ($n_in images)"

  # -npp/-nps 2 keeps preprocessing/segmentation worker memory modest while the
  # U-Net baseline is still training on the same GPU.
  if "$BIN/nnUNetv2_predict" -i "$src" -o "$dst" -d 1 -c 2d -f 0 \
       -chk checkpoint_best.pth --disable_tta -npp 2 -nps 2; then
    n_out=$(ls "$dst"/*.nii.gz 2>/dev/null | wc -l)
    echo "$(ts)  $name done: $n_out / $n_in masks"
    [ "$n_out" -eq "$n_in" ] || { echo "$(ts)  MISMATCH for $name"; fail=1; }
  else
    echo "$(ts)  $name FAILED (exit $?)"; fail=1
  fi
done

if [ "$fail" = 0 ]; then
  touch "$OUT/.complete"
  echo "$(ts)  ALL COHORTS COMPLETE -> $OUT"
  echo "$(ts)  next: seg_metrics_engine.py + make_table2.py, then"
  echo "$(ts)        export_locked_predictions.py for the automatic-mask variant"
else
  echo "$(ts)  FINISHED WITH FAILURES"
fi
