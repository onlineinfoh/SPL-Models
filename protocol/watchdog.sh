#!/usr/bin/env bash
# Overnight watchdog for the segmentation rebuild.
#
# Does three things every 10 minutes:
#   1. Appends a status line to protocol/watchdog.log
#   2. Flags a job that has stopped without reaching its target
#   3. When nnU-Net training completes, runs mask prediction for all four
#      cohorts automatically, so that step is not waiting for a human
#
# Predictions are written to seg-model-training/nnunet/predicted_masks_v2/,
# a NEW directory. Nothing existing is overwritten; the published *_model mask
# directories under data/ are left untouched.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export nnUNet_raw="$REPO/seg-model-training/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$REPO/seg-model-training/nnunet/nnUNet_preprocessed"
export nnUNet_results="$REPO/seg-model-training/nnunet/nnUNet_results"
BIN="$HOME/venvs/prism/bin"

LOG="$REPO/protocol/watchdog.log"
PRED_ROOT="$REPO/seg-model-training/nnunet/predicted_masks_v2"
PRED_DONE="$PRED_ROOT/.complete"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" >> "$LOG"; }

nnunet_epochs() {
  local f
  f=$(find "$nnUNet_results/Dataset001_lungval" -name 'training_log_*.txt' 2>/dev/null | head -1)
  [ -n "$f" ] && grep -ac 'Epoch time' "$f" 2>/dev/null || echo 0
}
unet_epochs() { grep -c '^[0-9]' "$REPO/seg-model-training/unet_locked/train_log.txt" 2>/dev/null || echo 0; }
running()     { pgrep -f "$1" >/dev/null 2>&1; }

predict_masks() {
  [ -f "$PRED_DONE" ] && return 0
  log "nnU-Net training finished; starting mask prediction for four cohorts"
  mkdir -p "$PRED_ROOT"
  local ok=1
  for spec in "train:data/train/imagesTr" \
              "internal_val:data/val/img_v" \
              "external_test1:data/test1/img_test1" \
              "external_test2:data/test2/img_test2"; do
    local name="${spec%%:*}" src="$REPO/${spec##*:}" dst="$PRED_ROOT/$name"
    mkdir -p "$dst"
    log "  predicting $name from $src"
    if "$BIN/nnUNetv2_predict" -i "$src" -o "$dst" -d 1 -c 2d -f 0 \
         -chk checkpoint_best.pth --disable_tta >> "$LOG" 2>&1; then
      log "  $name done: $(ls "$dst"/*.nii.gz 2>/dev/null | wc -l) masks"
    else
      log "  $name FAILED (see above)"; ok=0
    fi
  done
  if [ "$ok" = 1 ]; then
    touch "$PRED_DONE"
    log "mask prediction COMPLETE -> $PRED_ROOT"
    log "NEXT: re-run seg_metrics_engine.py and make_table2.py against these masks,"
    log "      then export_locked_predictions.py for the automatic-mask variant"
  fi
}

log "watchdog started (pid $$)"
while true; do
  ne=$(nnunet_epochs); ue=$(unet_epochs)
  nr=$(running nnUNetv2_train && echo up || echo down)
  ur=$(running "bin/python seg-model-training/unet_train" && echo up || echo down)
  gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | tr -d '\n')
  disk=$(df -h "$REPO" | awk 'NR==2{print $4}')
  log "nnU-Net ${ne}/1000 [$nr]  U-Net ${ue}/350 [$ur]  GPU ${gpu}  free ${disk}"

  if [ "$nr" = down ]; then
    if [ "$ne" -ge 999 ]; then predict_masks
    elif [ "$ne" -gt 0 ]; then log "WARNING: nnU-Net stopped early at epoch ${ne}/1000"; fi
  fi
  if [ "$ur" = down ] && [ "$ue" -gt 0 ] && [ "$ue" -lt 350 ]; then
    log "WARNING: U-Net stopped early at epoch ${ue}/350"
  fi

  if [ "$nr" = down ] && [ "$ur" = down ] && [ -f "$PRED_DONE" ]; then
    log "all jobs finished and masks predicted; watchdog exiting"
    exit 0
  fi
  sleep 600
done
