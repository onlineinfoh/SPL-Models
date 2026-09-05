#!/usr/bin/env python3
"""
Quick visual check for axis swap: load one case, optionally transpose the mask to
match the image, and save a PNG overlay.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def load_array(path: Path) -> np.ndarray:
    arr = np.asarray(nib.load(str(path)).get_fdata())
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        # drop singleton channel if present
        if arr.shape[0] in (1, 3):
            arr = arr[0]
        elif arr.shape[-1] in (1, 3):
            arr = arr[..., 0]
    return arr


def align_mask_to_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Try to align mask axes to image. If swapping last two axes matches better, do it.
    Then crop both to common shape (min height/width) so we can overlay.
    """
    mask_aligned = mask
    if image.ndim == mask.ndim == 2:
        if image.shape != mask.shape and image.shape == mask.T.shape:
            print(f"[info] transposing mask from {mask.shape} to match image {image.shape}")
            mask_aligned = mask.T
    elif image.ndim == mask.ndim == 3 and image.shape[0] == mask.shape[0]:
        if image.shape[1:] != mask.shape[1:] and image.shape[1] == mask.shape[2] and image.shape[2] == mask.shape[1]:
            print(f"[info] transposing mask spatial dims from {mask.shape} to match image {image.shape}")
            mask_aligned = mask.transpose(0, 2, 1)
    return mask_aligned


def main():
    # Pick one val case; change these if you want a different example.
    case_id = "18_seg"
    data_root = Path(__file__).resolve().parents[2] / "data"
    img_path = data_root / "test1/img_test1" / f"case_00018_0000.nii.gz"
    mask_path = data_root / "test1/seg_test1" / f"{case_id}.nii.gz"

    if not img_path.exists() or not mask_path.exists():
        raise SystemExit(f"Missing image or mask: {img_path}, {mask_path}")

    image = load_array(img_path)
    mask = load_array(mask_path)
    mask = align_mask_to_image(image, mask)

    # crop to common shape for overlay
    h = min(image.shape[-2], mask.shape[-2])
    w = min(image.shape[-1], mask.shape[-1])
    img_c = image[:h, :w]
    mask_c = mask[:h, :w]
    mask_bin = (mask_c > 0.5).astype(float)

    # normalize image for display
    img_disp = img_c
    if img_disp.max() > img_disp.min():
        img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min())

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_disp, cmap="gray")
    axes[0].imshow(mask_bin, cmap="Reds", alpha=0.4)
    axes[0].set_title("Image + aligned mask")
    axes[0].axis("off")

    axes[1].imshow(mask_bin, cmap="Reds")
    axes[1].set_title(f"Mask (aligned, shape {mask_bin.shape})")
    axes[1].axis("off")

    out_path = Path(__file__).resolve().parent / f"overlay_{case_id}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved overlay to {out_path}")


if __name__ == "__main__":
    main()
