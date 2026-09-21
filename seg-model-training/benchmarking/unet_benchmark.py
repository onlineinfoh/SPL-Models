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
Run Pytorch-UNet checkpoint inference on multiple NIfTI splits using
the same normalization as training, but with flexible mask naming
(supports case_XXXXX and numeric *_seg masks).

Checkpoint:
  seg-model-training/Pytorch-UNet/checkpoints/checkpoint_best.pth

Datasets:
  val:   new_data/val/img_v       -> seg_v
  train: new_data/train/imagesTr  -> labelsTr
  test1: new_data/test1/img_test1 -> seg_test1 (e.g., 1_seg.nii.gz)
  test2: new_data/test2/img_test2 -> seg_test2
"""

from __future__ import annotations
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# add Pytorch-UNet repo to path
PROJECT_ROOT = Path(__file__).resolve().parents[1] / "Pytorch-UNet"
sys.path.append(str(PROJECT_ROOT))

from unet import UNet  # type: ignore  # noqa: E402
from evaluate import _binary_batch_metrics  # type: ignore  # noqa: E402

DATA_ROOT = PROJECT_ROOT.parents[1] / "data"


def _strip_state_prefix(state: dict) -> dict:
    if any(k.startswith("module.") for k in state.keys()):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def _infer_n_classes(state: dict) -> int | None:
    weight = state.get("outc.conv.weight")
    if weight is not None and hasattr(weight, "shape") and len(weight.shape) > 0:
        return int(weight.shape[0])
    bias = state.get("outc.conv.bias")
    if bias is not None and hasattr(bias, "shape") and len(bias.shape) > 0:
        return int(bias.shape[0])
    return None


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


class FlexibleNiftiDataset(Dataset):
    def __init__(self, img_dir: Path, mask_dir: Path, target_size: int | None = None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.target_size = target_size
        self.items: list[tuple[Path, Path, int]] = []

        img_paths = sorted([p for p in self.img_dir.iterdir() if p.is_file() and (p.suffix in (".nii", ".gz") or p.name.endswith(".nii.gz"))])
        for img_path in img_paths:
            try:
                mask_path = find_mask_for_image(img_path, self.mask_dir)
            except FileNotFoundError:
                continue
            img_vol = load_nifti(img_path)
            mask_vol = load_nifti(mask_path)
            if img_vol.ndim > 3:
                img_vol = np.squeeze(img_vol)
            if mask_vol.ndim > 3:
                mask_vol = np.squeeze(mask_vol)
            if img_vol.ndim == 2:
                img_vol = img_vol[None, ...]
            if mask_vol.ndim == 2:
                mask_vol = mask_vol[None, ...]
            if img_vol.shape[1:] != mask_vol.shape[1:]:
                continue
            depth = min(img_vol.shape[0], mask_vol.shape[0])
            for z in range(depth):
                self.items.append((img_path, mask_path, z))
        if not self.items:
            raise RuntimeError(f"No NIfTI pairs found in {img_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path, z = self.items[idx]
        img_vol = load_nifti(img_path)
        mask_vol = load_nifti(mask_path)
        if img_vol.ndim > 3:
            img_vol = np.squeeze(img_vol)
        if mask_vol.ndim > 3:
            mask_vol = np.squeeze(mask_vol)
        if img_vol.ndim == 2:
            img_vol = img_vol[None, ...]
        if mask_vol.ndim == 2:
            mask_vol = mask_vol[None, ...]
        img = img_vol[z]
        mask = mask_vol[z]
        if img.max() > 0:
            img = img / img.max()
        mask = (mask > 0.5).astype(np.uint8)

        if self.target_size is not None:
            img_pil = Image.fromarray(img.astype(np.float32))
            mask_pil = Image.fromarray(mask.astype(np.float32))
            img_rs = img_pil.resize((self.target_size, self.target_size), resample=Image.BICUBIC)
            mask_rs = mask_pil.resize((self.target_size, self.target_size), resample=Image.NEAREST)
            img_arr = np.asarray(img_rs, dtype=np.float32)
            mask_arr = np.asarray(mask_rs, dtype=np.int64)
        else:
            img_arr = img.astype(np.float32)
            mask_arr = mask.astype(np.int64)

        img_arr = img_arr[None, ...]  # C,H,W
        return {
            "image": torch.from_numpy(np.ascontiguousarray(img_arr)).float(),
            "mask": torch.from_numpy(np.ascontiguousarray(mask_arr)).long(),
        }


def build_loader(
    img_dir: Path,
    mask_dir: Path,
    target_size: int | None = None,
    batch_size: int = 1,
) -> DataLoader:
    ds = FlexibleNiftiDataset(img_dir, mask_dir, target_size=target_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)


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


@torch.inference_mode()
def evaluate_with_ci(net, dataloader, device, amp):
    net.eval()
    series = {"Dice": [], "mIoU": [], "Precision": [], "Recall": [], "FPR": []}

    for batch in dataloader:
        image, mask_true = batch["image"], batch["mask"]
        image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        mask_true = mask_true.to(device=device, dtype=torch.long)

        with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
            mask_pred = net(image)

        if net.n_classes == 1:
            prob = torch.sigmoid(mask_pred)
            pred_mask = (prob > 0.5).float()
            pred_np = pred_mask.squeeze(1).cpu().numpy()
            true_np = mask_true.cpu().numpy()
            for p, t in zip(pred_np, true_np):
                dice, iou, prec, rec, fpr = _binary_batch_metrics(p[np.newaxis, ...], t[np.newaxis, ...])
                series["Dice"].append(float(dice))
                series["mIoU"].append(float(iou))
                series["Precision"].append(float(prec))
                series["Recall"].append(float(rec))
                series["FPR"].append(float(fpr))
        else:
            mask_pred_cls = mask_pred.argmax(dim=1)
            pred_bin = (mask_pred_cls == 1).cpu().numpy().astype(np.uint8)
            true_bin = (mask_true == 1).cpu().numpy().astype(np.uint8)
            for p, t in zip(pred_bin, true_bin):
                dice, iou, prec, rec, fpr = _binary_batch_metrics(p[np.newaxis, ...], t[np.newaxis, ...])
                series["Dice"].append(float(dice))
                series["mIoU"].append(float(iou))
                series["Precision"].append(float(prec))
                series["Recall"].append(float(rec))
                series["FPR"].append(float(fpr))

    net.train()
    return {k: _mean_ci95(v) for k, v in series.items()}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "checkpoint_best.pth"

    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict):
        state.pop("mask_values", None)
        state = _strip_state_prefix(state)
    n_classes = _infer_n_classes(state) or 2
    model = UNet(n_channels=1, n_classes=n_classes, bilinear=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    target_size = 512
    batch_size = 1

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

    results: dict[str, dict[str, tuple[float, float, float]]] = {}
    for name, (img_dir, mask_dir) in splits.items():
        loader = build_loader(img_dir, mask_dir, target_size=target_size, batch_size=batch_size)
        metrics = evaluate_with_ci(model, loader, device, amp=False)
        results[name] = metrics
        print(f"[{name}] {json.dumps(results[name], indent=2)}")

    out_dir = Path(__file__).resolve().parents[2] / "binary_classification" / "nnunet_benchmark_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.txt"
    with out_path.open("w") as f:
        for name, metrics in results.items():
            f.write(f"{name}:\n")
            for k, (mean, low, high) in metrics.items():
                f.write(f"  {k}: {mean:.4f}, 95%, {low:.4f}-{high:.4f}\n")
            f.write("\n")
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
