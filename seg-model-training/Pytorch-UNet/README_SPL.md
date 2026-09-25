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
- input size: 512 x 512 (`--size 512`)
- image scaling: 1.0 (`--scale`, per-image max normalisation)

`--size` must be passed explicitly; its default is `None`, which leaves slices at
native resolution.

## Train
```bash
python train.py \
  --images ../../data/train/imagesTr \
  --masks ../../data/train/labelsTr \
  --epochs 350 \
  --batch-size 1 \
  --learning-rate 5e-4 \
  --validation 0 \
  --classes 2 \
  --channels 1 \
  --scale 1.0 \
  --size 512
```

## Outputs
Checkpoints and logs are written under `checkpoints/` (excluded from git).
