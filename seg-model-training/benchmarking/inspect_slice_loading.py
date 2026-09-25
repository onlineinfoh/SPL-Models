#!/usr/bin/env python3
"""
Inspect how the training loader (NiftiSliceDataset) and the inference
loader (FlexibleNiftiDataset) slice NIfTI volumes. Prints the raw NIfTI
shape and the first slice shape each loader would feed the U-Net.

Usage example:
  python seg-model-training/benchmarking/inspect_slice_loading.py \
    --img-dir data/train/imagesTr --mask-dir data/train/labelsTr \
    --target-size 800

This does not load a model; it only inspects dataset behavior.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import nibabel as nib
import numpy as np


# Make Pytorch-UNet code importable
PROJECT_ROOT = Path(__file__).resolve().parents[1] / "Pytorch-UNet"
sys.path.append(str(PROJECT_ROOT))

from utils.data_loading import NiftiSliceDataset  # type: ignore  # noqa: E402


# Make the benchmarking loader importable
BENCH_DIR = Path(__file__).resolve().parent
sys.path.append(str(BENCH_DIR))
from unet_benchmark import FlexibleNiftiDataset  # type: ignore  # noqa: E402


def find_first_image(img_dir: Path) -> Path:
    imgs = sorted([p for p in img_dir.glob("*.nii*") if p.is_file()])
    if not imgs:
        raise FileNotFoundError(f"No NIfTI files found in {img_dir}")
    return imgs[0]


def main():
    parser = argparse.ArgumentParser(description="Inspect slice orientation for training vs inference loaders")
    parser.add_argument("--img-dir", type=Path, required=True, help="Directory with image NIfTI files")
    parser.add_argument("--mask-dir", type=Path, required=True, help="Directory with mask NIfTI files")
    parser.add_argument("--target-size", type=int, default=800, help="Square resize used in your runs")
    args = parser.parse_args()

    img_path = find_first_image(args.img_dir)
    print(f"Using image: {img_path}")

    nii = nib.load(str(img_path))
    data = np.asarray(nii.get_fdata())
    print(f"Raw NIfTI shape: {data.shape}, dtype: {data.dtype}")

    # Training loader
    train_ds = NiftiSliceDataset(str(args.img_dir), str(args.mask_dir), target_size=args.target_size)
    train_z = train_ds.items[0][2] if hasattr(train_ds, "items") else 0
    train_sample = train_ds[0]
    print("\nTraining loader (NiftiSliceDataset):")
    print(f"  depth reported: {len(train_ds)} slices")
    print(f"  first slice index z={train_z} uses img_vol[z]")
    print(f"  sample image tensor shape: {tuple(train_sample['image'].shape)}")
    print(f"  sample mask tensor shape:  {tuple(train_sample['mask'].shape)}")

    # Show what that raw slice looked like before resizing/normalizing
    try:
        raw_train_slice = data[train_z]
        print(f"  raw slice shape via img_vol[z]: {raw_train_slice.shape}")
    except Exception as e:  # pragma: no cover - best effort
        print(f"  could not extract raw train slice: {e}")

    # Inference loader
    inf_ds = FlexibleNiftiDataset(args.img_dir, args.mask_dir, target_size=args.target_size)
    inf_z = inf_ds.items[0][2] if hasattr(inf_ds, "items") else 0
    inf_sample = inf_ds[0]
    print("\nInference loader (FlexibleNiftiDataset):")
    print(f"  depth reported: {len(inf_ds)} slices")
    print(f"  first slice index z={inf_z} uses img_vol[:, :, z]")
    print(f"  sample image tensor shape: {tuple(inf_sample['image'].shape)}")
    print(f"  sample mask tensor shape:  {tuple(inf_sample['mask'].shape)}")

    try:
        raw_inf_slice = data[:, :, inf_z]
        print(f"  raw slice shape via img_vol[:, :, z]: {raw_inf_slice.shape}")
    except Exception as e:  # pragma: no cover - best effort
        print(f"  could not extract raw inference slice: {e}")


if __name__ == "__main__":
    main()
