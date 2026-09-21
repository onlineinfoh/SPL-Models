#!/usr/bin/env bash
# Waits for U-Net training to finish, then runs every remaining rebuild step.
#
#   1. U-Net predicted masks from the retrained checkpoint
#   2. Segmentation metrics across all three retrained arms
#   3. Table 2
#   4. Classification automatic-mask variant, re-exported against the NEW
#      nnU-Net masks (the earlier export used the superseded *_model masks)
#   5. Downstream statistics on the corrected variant
#   6. Post-hoc calibration, and task3/task4 on the calibrated scale
#
# Every step writes under protocol/results/ or analysis/masks_locked_v2/.
# Nothing superseded is overwritten.
#
# Run detached:
#   setsid nohup bash protocol/finalize_rebuild.sh \
#     > protocol/finalize.log 2>&1 < /dev/null &
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="$HOME/venvs/prism/bin/python"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
step() { echo; echo "=========== $(ts)  $* ==========="; }
fail=0
run() { "$@" || { echo "$(ts)  STEP FAILED: $*"; fail=1; }; }

# ---- 0. wait for U-Net ----------------------------------------------------
step "waiting for U-Net to finish"
while pgrep -f "bin/python seg-model-training/unet_train" >/dev/null 2>&1; do sleep 120; done
EP=$(grep -c '^[0-9]' seg-model-training/unet_locked/train_log.txt 2>/dev/null || echo 0)
echo "$(ts)  U-Net stopped at epoch ${EP}/350"
BEST=$(awk 'NR>4 && /^[0-9]/ && $8!="" {if($8+0>m){m=$8+0;e=$1}} END{print m" @ epoch "e}' \
       seg-model-training/unet_locked/train_log.txt)
echo "$(ts)  best held-out Dice: ${BEST}"
[ -f seg-model-training/unet_locked/checkpoint_best.pth ] || {
  echo "$(ts)  ABORT: no U-Net checkpoint"; exit 1; }

# ---- 1. U-Net masks -------------------------------------------------------
step "1/6  U-Net predicted masks"
export SPL_MASKS_OUT="$REPO/analysis/masks_locked_v2"
export SPL_UNET_CKPT="$REPO/seg-model-training/unet_locked/checkpoint_best.pth"
run "$PY" analysis/code/run_seg_inference.py --models unet
echo "$(ts)  U-Net masks: $(find analysis/masks_locked_v2/UNet -name '*.nii.gz' 2>/dev/null | wc -l)"

# ---- 2/3. segmentation metrics + Table 2 ----------------------------------
step "2/6  segmentation metrics (all three retrained arms)"
export SPL_NNUNET_PRED="$REPO/seg-model-training/nnunet/predicted_masks_v2"
export SPL_MASKS_LOCKED="$REPO/analysis/masks_locked_v2"
export SPL_RESULTS="$REPO/protocol/results/analysis"
mkdir -p "$SPL_RESULTS"
run "$PY" analysis/code/seg_metrics_engine.py

step "3/6  Table 2"
run "$PY" analysis/code/make_table2.py

# ---- 4. classification model variant against the NEW masks ----------------
step "4/6  re-export classification predictions (new nnU-Net masks)"
run "$PY" analysis/code/export_locked_predictions.py \
    --nnunet-masks "$REPO/seg-model-training/nnunet/predicted_masks_v2" \
    --out "$REPO/protocol/results/predictions_locked"

# ---- 5. downstream statistics ---------------------------------------------
step "5/6  downstream statistics on the corrected variant"
export SPL_PRED_DIR="$REPO/protocol/results/predictions_locked"
export SPL_ARCH=efficientnet_b0
export SPL_FIGURES="$REPO/protocol/results/figures"
export SPL_SEG_METRICS="$SPL_RESULTS/seg_metrics_per_case.csv"
mkdir -p "$SPL_FIGURES"
for t in task1_confidence_intervals task2_threshold_policy task3_calibration \
         task4_dca task6_gt_vs_model_mask subgroup_and_precision; do
  run "$PY" "analysis/code/$t.py" >/dev/null && echo "$(ts)  ok  $t"
done

# ---- 6. calibration, and task3/4 on the calibrated scale ------------------
step "6/6  calibration"
run "$PY" analysis/code/calibrate_locked_model.py
export SPL_PRED_DIR="$REPO/protocol/results/predictions_calibrated"
export SPL_RESULTS="$REPO/protocol/results/analysis_calibrated"
export SPL_FIGURES="$REPO/protocol/results/figures_calibrated"
mkdir -p "$SPL_RESULTS" "$SPL_FIGURES"
for t in task3_calibration task4_dca; do
  run "$PY" "analysis/code/$t.py" >/dev/null && echo "$(ts)  ok  $t (calibrated)"
done

step "SUMMARY"
if [ "$fail" = 0 ]; then echo "$(ts)  REBUILD COMPLETE, no failures"
else echo "$(ts)  REBUILD FINISHED WITH FAILURES (search for 'STEP FAILED')"; fi
echo "$(ts)  Table 2: protocol/results/analysis/table2_corrected.md"
