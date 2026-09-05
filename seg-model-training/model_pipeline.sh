#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -euo pipefail

PYTHON="${PYTHON:-python3}"

TRAIN_SCRIPT="$REPO_ROOT/seg-model-training/Pytorch-UNet/train.py"
BENCH_SCRIPT="$REPO_ROOT/seg-model-training/benchmarking/unet_benchmark.py"

# "${PYTHON}" "${TRAIN_SCRIPT}" \
#   --epochs 450 \
#   --batch-size  4 \
#   --scale 1.0 \
#   --size 512 \
#   --learning-rate 5e-4 \
#   --validation 0 \
#   --images $REPO_ROOT/data/train/imagesTr \
#   --masks $REPO_ROOT/data/train/labelsTr

"${PYTHON}" "${BENCH_SCRIPT}"

# "${PYTHON}" $REPO_ROOT/seg-model-training/DeepLabV3Plus-Pytorch/main.py \
#   --dataset lung \
#   --lung_img_dir $REPO_ROOT/data/train/imagesTr \
#   --lung_mask_dir $REPO_ROOT/data/train/labelsTr \
#   --total_itrs 100000 \
#   --crop_size 512 \
#   --batch_size 8 \
#   --val_batch_size 8

"${PYTHON}" $REPO_ROOT/seg-model-training/benchmarking/deeplab_benchmark.py
