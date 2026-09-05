#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"

TRAIN_SCRIPT="/home/tianxi-liang/TianxiLiang/research/china/seg-model-training/Pytorch-UNet/train.py"
BENCH_SCRIPT="/home/tianxi-liang/TianxiLiang/research/china/seg-model-training/benchmarking/unet_benchmark.py"

# "${PYTHON}" "${TRAIN_SCRIPT}" \
#   --epochs 450 \
#   --batch-size  4 \
#   --scale 1.0 \
#   --size 512 \
#   --learning-rate 5e-4 \
#   --validation 0 \
#   --images /home/tianxi-liang/TianxiLiang/research/china/new_data/train/imagesTr \
#   --masks /home/tianxi-liang/TianxiLiang/research/china/new_data/train/labelsTr

"${PYTHON}" "${BENCH_SCRIPT}"

# "${PYTHON}" /home/tianxi-liang/TianxiLiang/research/china/seg-model-training/DeepLabV3Plus-Pytorch/main.py \
#   --dataset lung \
#   --lung_img_dir /home/tianxi-liang/TianxiLiang/research/china/new_data/train/imagesTr \
#   --lung_mask_dir /home/tianxi-liang/TianxiLiang/research/china/new_data/train/labelsTr \
#   --total_itrs 100000 \
#   --crop_size 512 \
#   --batch_size 8 \
#   --val_batch_size 8

"${PYTHON}" /home/tianxi-liang/TianxiLiang/research/china/seg-model-training/benchmarking/deeplab_benchmark.py
