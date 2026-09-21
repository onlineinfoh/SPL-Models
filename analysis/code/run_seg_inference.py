"""
Write predicted segmentation masks for DeepLabv3+ and U-Net as NIfTI.

Both models run at 512x512 with their own original preprocessing. Predictions
are resampled back to the native ground-truth resolution with nearest-neighbour
interpolation before being written, so that seg_metrics_engine.py scores all
three models on the same grid as nnU-Net. This matters for FPR in particular,
whose denominator is the number of true background pixels and therefore depends
on the grid.

Existing weights are loaded as-is; nothing is retrained here.

Usage
-----
    ~/venvs/prism/bin/python analysis/code/run_seg_inference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
SEG_ROOT = REPO / "seg-model-training"
DL_ROOT = SEG_ROOT / "DeepLabV3Plus-Pytorch"
UNET_ROOT = SEG_ROOT / "Pytorch-UNet"
import os
# Output root and checkpoints are overridable so a retrained model can be
# written to a NEW directory. Defaults reproduce the superseded masks, which
# are the only remaining record of the original DeepLabv3+ weights (the
# retrain overwrote best_deeplabv3plus_mobilenet_lung_os16.pth in place).
OUT_ROOT = Path(os.environ.get("SPL_MASKS_OUT", REPO / "analysis" / "masks_locked"))

# DeepLabv3+ was trained with an explicit --crop_size 512, so it must be scored
# at 512. U-Net was NOT: Pytorch-UNet's --size defaults to None and the
# published training command omits it, so U-Net trained at native resolution.
# Scoring the U-Net at 512 produces near-empty masks (verified: Dice 0.7400 at
# native versus 0.0001 with 38/40 empty at 512 on the same checkpoint).
DEEPLAB_CROP_SIZE = 512
UNET_CROP_SIZE = None            # None = native resolution
CROP_SIZE = DEEPLAB_CROP_SIZE    # retained for the DeepLabv3+ path

COHORTS = {
    "train": ("data/train/imagesTr", "data/train/labelsTr"),
    "internal_val": ("data/val/img_v", "data/val/seg_v"),
    "external_test1": ("data/test1/img_test1", "data/test1/seg_test1"),
    "external_test2": ("data/test2/img_test2", "data/test2/seg_test2"),
}


def case_id_of(path: Path) -> str:
    base = path.name
    if base.endswith(".nii.gz"):
        base = base[:-7]
    elif base.endswith(".nii"):
        base = base[:-4]
    if base.endswith("_0000"):
        base = base[:-5]
    if base.endswith("_seg"):
        base = base[:-4]
    if base.startswith("case_"):
        return base
    digits = "".join(ch for ch in base if ch.isdigit())
    return f"case_{int(digits):05d}" if digits else f"case_{base}"


def find_mask(case: str, mask_dir: Path) -> Path | None:
    num = int("".join(ch for ch in case if ch.isdigit()))
    for cand in (mask_dir / f"{case}.nii.gz", mask_dir / f"{case}.nii",
                 mask_dir / f"{num}_seg.nii.gz", mask_dir / f"{num}_seg.nii",
                 mask_dir / f"{num}.nii.gz"):
        if cand.exists():
            return cand
    return None


def load_nifti_2d(path: Path) -> np.ndarray:
    arr = np.asarray(nib.load(str(path)).get_fdata())
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        arr = arr[0] if arr.shape[0] == 1 else arr[..., 0]
    return arr


def resize_pred_to_native(pred512: np.ndarray, native_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour upsample of a binary mask back to native resolution."""
    pil = Image.fromarray(pred512.astype(np.uint8), mode="L")
    # PIL size is (width, height) = (cols, rows)
    out = pil.resize((native_shape[1], native_shape[0]), resample=Image.NEAREST)
    return (np.asarray(out) > 0).astype(np.uint8)


def save_mask(mask: np.ndarray, ref_path: Path, out_path: Path) -> None:
    ref = nib.load(str(ref_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(mask.astype(np.uint8), ref.affine, ref.header)
    img.set_data_dtype(np.uint8)
    nib.save(img, str(out_path))


# --------------------------------------------------------------------------
# DeepLabv3+ (mobilenet backbone, output_stride 16)
# --------------------------------------------------------------------------
def run_deeplab(device: torch.device) -> None:
    sys.path.extend([str(SEG_ROOT), str(DL_ROOT), str(DL_ROOT / "utils")])
    import network.modeling as modeling  # type: ignore

    ckpt_path = Path(os.environ.get("SPL_DEEPLAB_CKPT",
        DL_ROOT / "checkpoints" / "best_deeplabv3plus_mobilenet_lung_os16.pth"))
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = modeling.deeplabv3plus_mobilenet(num_classes=2, output_stride=16,
                                             pretrained_backbone=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"[DeepLabv3+] loaded {ckpt_path.name}")

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for cohort, (img_rel, gt_rel) in COHORTS.items():
        img_dir, gt_dir = REPO / img_rel, REPO / gt_rel
        out_dir = OUT_ROOT / "DeepLabv3plus" / cohort
        n = 0
        for img_path in sorted(img_dir.glob("*.nii.gz")):
            case = case_id_of(img_path)
            gt_path = find_mask(case, gt_dir)
            if gt_path is None:
                continue
            native = load_nifti_2d(gt_path).shape

            # same preprocessing as deeplab_benchmark.py
            arr = load_nifti_2d(img_path)
            arr = arr - arr.min()
            if arr.max() > 0:
                arr = arr / arr.max()
            u8 = (arr * 255.0).clip(0, 255).astype(np.uint8)
            rgb = np.repeat(u8[..., None], 3, axis=2)
            pil = Image.fromarray(rgb).resize((CROP_SIZE, CROP_SIZE),
                                              resample=Image.BILINEAR)
            x = np.asarray(pil, dtype=np.float32) / 255.0
            x = (x - mean) / std
            t = torch.from_numpy(x.transpose(2, 0, 1))[None].to(device)

            with torch.no_grad():
                logits = model(t)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

            save_mask(resize_pred_to_native(pred, native), gt_path,
                      out_dir / f"{case}.nii.gz")
            n += 1
        print(f"  {cohort:15s} wrote {n} masks -> {out_dir}")


# --------------------------------------------------------------------------
# U-Net
# --------------------------------------------------------------------------
def run_unet(device: torch.device) -> None:
    sys.path.append(str(UNET_ROOT))
    from unet import UNet  # type: ignore

    ckpt_path = Path(os.environ.get("SPL_UNET_CKPT",
        UNET_ROOT / "checkpoints" / "checkpoint_best.pth"))
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    state.pop("mask_values", None)
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    w = state.get("outc.conv.weight")
    n_classes = int(w.shape[0]) if w is not None else 2

    model = UNet(n_channels=1, n_classes=n_classes, bilinear=False)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"[U-Net] loaded {ckpt_path.name} (n_classes={n_classes})")

    for cohort, (img_rel, gt_rel) in COHORTS.items():
        img_dir, gt_dir = REPO / img_rel, REPO / gt_rel
        out_dir = OUT_ROOT / "UNet" / cohort
        n = 0
        for img_path in sorted(img_dir.glob("*.nii.gz")):
            case = case_id_of(img_path)
            gt_path = find_mask(case, gt_dir)
            if gt_path is None:
                continue
            native = load_nifti_2d(gt_path).shape

            # same preprocessing as unet_benchmark.py
            arr = load_nifti_2d(img_path)
            if arr.max() > 0:
                arr = arr / arr.max()
            if UNET_CROP_SIZE is None:
                x = arr.astype(np.float32)[None, None]
            else:
                pil = Image.fromarray(arr.astype(np.float32))
                rs = pil.resize((UNET_CROP_SIZE, UNET_CROP_SIZE), resample=Image.BICUBIC)
                x = np.asarray(rs, dtype=np.float32)[None, None]
            t = torch.from_numpy(np.ascontiguousarray(x)).float().to(device)

            with torch.no_grad():
                out = model(t)
                if n_classes == 1:
                    pred = (torch.sigmoid(out) > 0.5).float()[0, 0].cpu().numpy()
                else:
                    pred = (out.argmax(dim=1)[0] == 1).cpu().numpy()
            pred = pred.astype(np.uint8)

            save_mask(resize_pred_to_native(pred, native), gt_path,
                      out_dir / f"{case}.nii.gz")
            n += 1
        print(f"  {cohort:15s} wrote {n} masks -> {out_dir}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["deeplab", "unet"],
                    choices=["deeplab", "unet"],
                    help="which baselines to predict; lets DeepLab run while "
                         "U-Net is still training")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    if "deeplab" in args.models:
        run_deeplab(device)
    if "unet" in args.models:
        run_unet(device)
    print("\nLocked masks written under", OUT_ROOT)


if __name__ == "__main__":
    main()
