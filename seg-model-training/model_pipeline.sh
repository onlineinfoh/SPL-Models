#!/usr/bin/env bash
set -euo pipefail

# Stage 1 baselines. The training calls are commented out because the reported
# numbers come from the checkpoints already present in each subdirectory.
# Parameters match Pytorch-UNet/README_SPL.md and
# DeepLabV3Plus-Pytorch/README_SPL.md.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON="${PYTHON:-python3}"

TRAIN_SCRIPT="$REPO_ROOT/seg-model-training/Pytorch-UNet/train.py"
BENCH_SCRIPT="$REPO_ROOT/seg-model-training/benchmarking/unet_benchmark.py"

# "${PYTHON}" "${TRAIN_SCRIPT}" \
#   --images "$REPO_ROOT/data/train/imagesTr" \
#   --masks "$REPO_ROOT/data/train/labelsTr" \
#   --epochs 350 \
#   --batch-size 1 \
#   --learning-rate 5e-4 \
#   --validation 0 \
#   --classes 2 \
#   --channels 1 \
#   --scale 1.0 \
#   --size 512

"${PYTHON}" "${BENCH_SCRIPT}"

# "${PYTHON}" "$REPO_ROOT/seg-model-training/DeepLabV3Plus-Pytorch/main.py" \
#   --dataset lung \
#   --lung_img_dir "$REPO_ROOT/data/train/imagesTr" \
#   --lung_mask_dir "$REPO_ROOT/data/train/labelsTr" \
#   --num_classes 2 \
#   --crop_size 512 \
#   --batch_size 4 \
#   --val_interval 200 \
#   --total_itrs 30000

"${PYTHON}" "$REPO_ROOT/seg-model-training/benchmarking/deeplab_benchmark.py"
