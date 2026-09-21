#!/usr/bin/env python3
# ==========================================================================
# SUPERSEDED BY THE LOCKED RE-ANALYSIS
# ==========================================================================
#
# One of three per-model benchmark scripts that do not share a metric
# definition. Replaced by analysis/code/seg_metrics_engine.py.
#
# Retained unmodified as the audit record. Produces no reported result.
# See README.md and docs/REPRODUCE.md for the active pipeline.
# ==========================================================================
#
"""
Benchmark nnU-Net checkpoint on the same datasets as benchmarking/unet_benchmark.py,
using nnU-Net's own preprocessing (dynamic resizing) and metrics aligned with the
Pytorch-UNet evaluation (Dice, mIoU, Precision, Recall, FPR).

Checkpoint:
  seg-model-training/nnunet/nnUNet_results/Dataset000_lung/nnUNetTrainer__nnUNetPlans__2d/fold_all/checkpoint_best.pth

Datasets:
  val:   new_data/val/img_v       -> seg_v
  train: new_data/train/imagesTr  -> labelsTr
  test1: new_data/test1/img_test1 -> seg_test1
  test2: new_data/test2/img_test2 -> seg_test2
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np
import torch

# Configure nnU-Net paths (needed even when using absolute model paths)
REPO_ROOT = Path(__file__).resolve().parents[1]  # seg-model-training
NNUNET_DATA_ROOT = REPO_ROOT / "nnunet"
os.environ.setdefault("nnUNet_raw", str(NNUNET_DATA_ROOT / "nnUNet_raw"))
os.environ.setdefault("nnUNet_preprocessed", str(NNUNET_DATA_ROOT / "nnUNet_preprocessed"))
os.environ.setdefault("nnUNet_results", str(NNUNET_DATA_ROOT / "nnUNet_results"))

# add nnUNet repo to path for nnunetv2 imports
NNUNET_REPO = Path(__file__).resolve().parents[2] / "nnUNet"
sys.path.append(str(NNUNET_REPO))

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # type: ignore  # noqa: E402

DATA_ROOT = REPO_ROOT.parent / "data"


def load_nifti(path: Path) -> np.ndarray:
    arr = np.asarray(nib.load(str(path)).get_fdata())
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return arr


def find_mask_for_image(img_path: Path, mask_dir: Path) -> Path:
    name = img_path.name
    base = name
    if base.endswith(".nii.gz"):
        base = base[:-7]
    elif base.endswith(".nii"):
        base = base[:-4]
    if base.endswith("_0000"):
        base = base[:-5]

    candidates = [
        mask_dir / f"{base}.nii.gz",
        mask_dir / f"{base}.nii",
    ]
    digits = "".join(ch for ch in base if ch.isdigit())
    if digits:
        num = int(digits)
        candidates += [
            mask_dir / f"{num}_seg.nii.gz",
            mask_dir / f"{num}_seg.nii",
            mask_dir / f"{num}.nii.gz",
            mask_dir / f"{num}.nii",
        ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"No mask found for {img_path.name}")


def _binary_batch_metrics(pred: np.ndarray, true: np.ndarray):
    """
    pred, true: shape (N, H, W), values {0,1}
    """
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0).astype(np.uint8)
    true = np.nan_to_num(true, nan=0.0, posinf=0.0, neginf=0.0).astype(np.uint8)
    tp = np.logical_and(pred == 1, true == 1).sum()
    fp = np.logical_and(pred == 1, true == 0).sum()
    fn = np.logical_and(pred == 0, true == 1).sum()
    tn = np.logical_and(pred == 0, true == 0).sum()
    denom_dice = (2 * tp + fp + fn + 1e-8)
    denom_iou = (tp + fp + fn + 1e-8)
    denom_prec = (tp + fp + 1e-8)
    denom_rec = (tp + fn + 1e-8)
    denom_fpr = (fp + tn + 1e-8)
    dice = (2 * tp) / denom_dice
    iou = tp / denom_iou
    prec = tp / denom_prec
    rec = tp / denom_rec
    fpr = fp / denom_fpr
    return dice, iou, prec, rec, fpr


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(mask), nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.squeeze(arr)
    return arr


def _binarize(mask: np.ndarray) -> np.ndarray:
    return (mask > 0.5).astype(np.uint8)


def _slice_mask(mask: np.ndarray) -> List[np.ndarray]:
    arr = _normalize_mask(mask)
    if arr.ndim == 2:
        return [_binarize(arr)]
    if arr.ndim == 3:
        # Heuristic: treat the smallest axis as depth if it looks like a slice/channel axis
        if arr.shape[0] <= min(arr.shape[1], arr.shape[2]):
            return [_binarize(arr[idx, ...]) for idx in range(arr.shape[0])]
        return [_binarize(arr[..., idx]) for idx in range(arr.shape[2])]
    raise ValueError(f"Unsupported mask dimensions: {arr.shape}")


def _maybe_align_shapes(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """
    Try to reconcile axis ordering (e.g., HxW vs WxH) before slicing.
    """
    pred_s = np.squeeze(pred)
    true_s = np.squeeze(true)
    if pred_s.shape == true_s.shape:
        return true_s
    # swap last two axes if that matches
    if pred_s.ndim == true_s.ndim and pred_s.shape[:-2] == true_s.shape[:-2]:
        if pred_s.shape[-2] == true_s.shape[-1] and pred_s.shape[-1] == true_s.shape[-2]:
            print(f"[WARN] Transposing ground truth from shape {true_s.shape} to match prediction {pred_s.shape}")
            axes = list(range(true_s.ndim))
            axes[-2], axes[-1] = axes[-1], axes[-2]
            return true_s.transpose(axes)
    # handle simple 2D swap
    if pred_s.ndim == true_s.ndim == 2 and pred_s.shape[0] == true_s.shape[1] and pred_s.shape[1] == true_s.shape[0]:
        print(f"[WARN] Transposing ground truth from shape {true_s.shape} to match prediction {pred_s.shape}")
        return true_s.T
    return true_s


def predict_segmentation(
    predictor: nnUNetPredictor,
    image_path: Path,
) -> np.ndarray:
    """
    Run prediction for a single image and return the predicted segmentation.
    """
    reader = predictor.plans_manager.image_reader_writer_class()
    image_np, image_props = reader.read_images((str(image_path),))
    pred_seg = predictor.predict_single_npy_array(image_np, image_props, None, None, False)
    return np.asarray(pred_seg)


def evaluate_split(
    predictor: nnUNetPredictor,
    img_dir: Path,
    mask_dir: Path,
) -> Dict[str, tuple[float, float, float]]:
    img_paths = sorted(
        [p for p in Path(img_dir).iterdir() if p.is_file() and (p.suffix in {".nii", ".gz"} or p.name.endswith(".nii.gz"))]
    )
    series = {"Dice": [], "mIoU": [], "Precision": [], "Recall": [], "FPR": []}

    for img_path in img_paths:
        try:
            mask_path = find_mask_for_image(img_path, mask_dir)
        except FileNotFoundError as e:
            print(f"[WARN] {e}")
            continue

        pred_vol = predict_segmentation(predictor, img_path)
        true_vol = load_nifti(mask_path)
        true_vol = _maybe_align_shapes(pred_vol, true_vol)

        try:
            pred_slices = _slice_mask(pred_vol)
            true_slices = _slice_mask(true_vol)
        except ValueError as e:
            print(f"[WARN] {img_path.name}: {e}")
            continue

        if len(pred_slices) != len(true_slices):
            print(f"[WARN] Slice count mismatch for {img_path.name}: pred {len(pred_slices)}, true {len(true_slices)}")
            continue

        for pred_slice, true_slice in zip(pred_slices, true_slices):
            if pred_slice.shape != true_slice.shape:
                if pred_slice.shape == true_slice.T.shape:
                    print(f"[WARN] Transposing slice for {img_path.name} from {true_slice.shape} to {pred_slice.shape}")
                    true_slice = true_slice.T
                else:
                    print(f"[WARN] Skipping slice for {img_path.name} due to irreconcilable shape mismatch: pred {pred_slice.shape}, true {true_slice.shape}")
                    continue
            dice, iou, prec, rec, fpr = _binary_batch_metrics(
                pred_slice[np.newaxis, ...],
                true_slice[np.newaxis, ...],
            )
            series["Dice"].append(float(dice))
            series["mIoU"].append(float(iou))
            series["Precision"].append(float(prec))
            series["Recall"].append(float(rec))
            series["FPR"].append(float(fpr))

    def _mean_ci95(values: list[float]) -> tuple[float, float, float]:
        if not values:
            return 0.0, 0.0, 0.0
        arr = np.asarray(values, dtype=np.float64)
        mean = float(arr.mean())
        if arr.size < 2:
            return mean, mean, mean
        se = float(arr.std(ddof=1) / np.sqrt(arr.size))
        delta = 1.96 * se
        return mean, mean - delta, mean + delta

    return {k: _mean_ci95(v) for k, v in series.items()}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_root = REPO_ROOT / "nnunet" / "nnUNet_results" / "Dataset000_lung" / "nnUNetTrainer__nnUNetPlans__2d"
    checkpoint_name = "checkpoint_best.pth"

    predictor = nnUNetPredictor(device=device, verbose=True)
    predictor.initialize_from_trained_model_folder(
        str(model_root),
        use_folds=("all",),
        checkpoint_name=checkpoint_name,
    )

    splits = {
        "val": (
            DATA_ROOT / "val/img_v",
            DATA_ROOT / "val/seg_v",
        ),
        "train": (
            DATA_ROOT / "train/imagesTr",
            DATA_ROOT / "train/labelsTr",
        ),
        "external_test1": (
            DATA_ROOT / "test1/img_test1",
            DATA_ROOT / "test1/seg_test1",
        ),
        "external_test2": (
            DATA_ROOT / "test2/img_test2",
            DATA_ROOT / "test2/seg_test2",
        ),
    }

    results: Dict[str, Dict[str, tuple[float, float, float]]] = {}
    for name, (img_dir, mask_dir) in splits.items():
        print(f"\n[{name}] Running nnU-Net inference on {img_dir} ...")
        metrics = evaluate_split(predictor, img_dir, mask_dir)
        results[name] = metrics
        print(f"[{name}] {json.dumps(results[name], indent=2)}")

    out_dir = Path(__file__).resolve().parents[2] / "binary_classification" / "nnunet_benchmark_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics_nnunet.txt"
    with out_path.open("w") as f:
        for name, metrics in results.items():
            f.write(f"{name}:\n")
            for k, (mean, low, high) in metrics.items():
                f.write(f"  {k}: {mean:.4f}, 95%, {low:.4f}-{high:.4f}\n")
            f.write("\n")
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
