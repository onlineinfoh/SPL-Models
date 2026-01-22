# SPL-Models Notes (Pytorch-UNet)

Original repository: https://github.com/milesial/Pytorch-UNet

## Data loading
Expected NIfTI layout (relative to repo root):
- `data/train/imagesTr` (images, e.g., `case_00001_0000.nii.gz`)
- `data/train/labelsTr` (masks, e.g., `case_00001.nii.gz`)

The training script uses `NiftiSliceDataset` to slice each 3D volume into 2D slices.

## Parameters used (paper run)
- model: `UNet` (2D)
- epochs: 350
- learning rate: 5e-4
- validation split: 0 (train set reused for validation metrics)
- classes: 2
- channels: 1
- batch size: 1

## Train
```bash
python train.py \
  --images ../../data/train/imagesTr \
  --masks ../../data/train/labelsTr \
  --epochs 350 \
  --learning-rate 5e-4 \
  --validation 0 \
  --classes 2 \
  --channels 1
```

## Outputs
Checkpoints and logs are written under `checkpoints/` (excluded from git).
