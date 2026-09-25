#!/usr/bin/env python3
"""
Benchmark trained DeepLabV3+ (mobilenet) checkpoint on the lung dataset splits.

Uses the same normalization/resize as validation in main.py (resize to crop_size,
to tensor, ImageNet mean/std), and computes binary metrics (Dice, IoU, Precision,
Recall, FPR) against ground-truth masks in data/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.serialization import add_safe_globals
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import sys
ROOT = Path(__file__).resolve().parents[1]
DL_ROOT = ROOT / "DeepLabV3Plus-Pytorch"
sys.path.extend([str(ROOT), str(DL_ROOT), str(DL_ROOT / "utils")])

import network.modeling as modeling  # type: ignore  # noqa: E402
import ext_transforms as et  # type: ignore  # noqa: E402
from metrics.stream_metrics import StreamSegMetrics  # type: ignore  # noqa: E402


CKPT_PATH = DL_ROOT / "checkpoints" / "best_deeplabv3plus_mobilenet_lung_os16.pth"
DATA_ROOT = ROOT.parent / "data"
CROP_SIZE = 512
BATCH_SIZE = 2

SPLITS = {
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


def load_nifti(path: Path) -> np.ndarray:
    arr = np.asarray(nib.load(str(path)).get_fdata())
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    arr = np.squeeze(arr)
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
            mask_dir / f"{num:05d}_seg.nii.gz",
            mask_dir / f"{num:05d}_seg.nii",
        ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"No mask found for {img_path.name}")


class LungEvalDataset(Dataset):
    def __init__(self, img_dir: Path, mask_dir: Path, transform):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.items: list[Tuple[Path, Path]] = []

        img_paths = sorted(
            [p for p in self.img_dir.iterdir() if p.is_file() and (p.suffix in (".nii", ".gz") or p.name.endswith(".nii.gz"))]
        )
        for img_path in img_paths:
            try:
                mask_path = find_mask_for_image(img_path, self.mask_dir)
            except FileNotFoundError:
                continue
            self.items.append((img_path, mask_path))
        if not self.items:
            raise RuntimeError(f"No NIfTI pairs found in {img_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path = self.items[idx]
        img_arr = load_nifti(img_path)
        mask_arr = load_nifti(mask_path)
        if mask_arr.shape != img_arr.shape:
            if mask_arr.shape[::-1] == img_arr.shape:
                mask_arr = mask_arr.T
            else:
                raise RuntimeError(f"Shape mismatch for {img_path.name}: img {img_arr.shape}, mask {mask_arr.shape}")

        # normalize image to 0-255 and repeat to 3 channels
        img = img_arr - img_arr.min()
        if img.max() > 0:
            img = img / img.max()
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        img_rgb = np.repeat(img[..., None], 3, axis=2)
        mask_bin = (mask_arr > 0.5).astype(np.uint8)

        img_pil = Image.fromarray(img_rgb)
        mask_pil = Image.fromarray(mask_bin, mode="L")
        img_t, mask_t = self.transform(img_pil, mask_pil)
        return {
            "image": img_t,
            "mask": torch.from_numpy(np.asarray(mask_t)).long(),
        }


def _binary_confusion_metrics(hist: np.ndarray) -> Dict[str, object]:
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / np.maximum(hist.sum(axis=1), 1e-8)
    iu = np.diag(hist) / np.maximum(hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist), 1e-8)
    freq = hist.sum(axis=1) / np.maximum(hist.sum(), 1e-8)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(hist.shape[0]), iu))

    tp = np.diag(hist)
    fp = hist.sum(axis=0) - tp
    fn = hist.sum(axis=1) - tp
    tn = hist.sum() - (tp + fp + fn)
    precision = tp / np.maximum(tp + fp, 1e-8)
    recall = tp / np.maximum(tp + fn, 1e-8)
    fpr = fp / np.maximum(fp + tn, 1e-8)
    dice = (2 * tp) / np.maximum(2 * tp + fp + fn, 1e-8)

    return {
        "Overall Acc": float(acc),
        "Mean Acc": float(np.nanmean(acc_cls)),
        "FreqW Acc": float(fwavacc),
        "Mean IoU": float(np.nanmean(iu)),
        "Class IoU": {int(k): float(v) for k, v in cls_iu.items()},
        "Mean Dice": float(np.nanmean(dice)),
        "Mean Precision": float(np.nanmean(precision)),
        "Mean Recall": float(np.nanmean(recall)),
        "Mean FPR": float(np.nanmean(fpr)),
    }


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


def evaluate_split(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, object]:
    scalar_series = {
        "Overall Acc": [],
        "Mean Acc": [],
        "FreqW Acc": [],
        "Mean IoU": [],
        "Mean Dice": [],
        "Mean Precision": [],
        "Mean Recall": [],
        "Mean FPR": [],
    }
    class_iou_series = {0: [], 1: []}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            preds = logits.argmax(dim=1)
            for true_mask, pred_mask in zip(masks.cpu().numpy(), preds.cpu().numpy()):
                hist = np.zeros((2, 2), dtype=np.float64)
                true_flat = true_mask.reshape(-1).astype(np.int64)
                pred_flat = pred_mask.reshape(-1).astype(np.int64)
                np.add.at(hist, (true_flat, pred_flat), 1)
                sample_metrics = _binary_confusion_metrics(hist)
                for key in scalar_series:
                    scalar_series[key].append(float(sample_metrics[key]))
                for cls, value in sample_metrics["Class IoU"].items():
                    class_iou_series[int(cls)].append(float(value))

    results: Dict[str, object] = {k: _mean_ci95(v) for k, v in scalar_series.items()}
    results["Class IoU"] = {cls: _mean_ci95(vals) for cls, vals in class_iou_series.items()}
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # allow legacy numpy scalar in checkpoint
    try:
        add_safe_globals([np.core.multiarray.scalar])  # type: ignore[attr-defined]
    except Exception:
        pass
    checkpoint = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model = modeling.deeplabv3plus_mobilenet(num_classes=2, output_stride=16, pretrained_backbone=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    val_transform = et.ExtCompose([
        et.ExtResize(size=(CROP_SIZE, CROP_SIZE)),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    results: Dict[str, Dict[str, object]] = {}
    for name, (img_dir, mask_dir) in SPLITS.items():
        ds = LungEvalDataset(img_dir, mask_dir, transform=val_transform)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)
        metrics = evaluate_split(model, loader, device)
        results[name] = metrics
        print(f"[{name}] {metrics}")

    out_dir = ROOT.parent / "binary_classification" / "nnunet_benchmark_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics_deeplab.txt"
    with out_path.open("w") as f:
        for name, metrics in results.items():
            f.write(f"{name}:\n")
            for k, v in metrics.items():
                if isinstance(v, dict):
                    f.write(f"  {k}:\n")
                    for kk, vv in v.items():
                        mean, low, high = vv
                        f.write(f"    {kk}: {mean:.4f}, 95%, {low:.4f}-{high:.4f}\n")
                else:
                    mean, low, high = v
                    f.write(f"  {k}: {mean:.4f}, 95%, {low:.4f}-{high:.4f}\n")
            f.write("\n")
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
