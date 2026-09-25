"""
Segmentation metrics for all three models, computed from the locked masks.

Every metric is computed on the lesion (foreground) class only, per image, and
then averaged across patients. There is one image per patient, so image-level
and patient-level averages coincide. Confidence intervals are patient-level
bootstrap percentile intervals.

    TP = |pred=1 AND gt=1|      FP = |pred=1 AND gt=0|
    FN = |pred=0 AND gt=1|      TN = |pred=0 AND gt=0|
    Dice      = 2TP / (2TP + FP + FN)
    IoU       = TP / (TP + FP + FN)
    Precision = TP / (TP + FP)        denominator = all pixels predicted lesion
    Recall    = TP / (TP + FN)        denominator = all pixels truly lesion
    FPR       = FP / (FP + TN)        denominator = all pixels truly background
    HD95      = 95th percentile of the symmetric boundary distance set
    ASSD      = mean of the symmetric boundary distance set

HD95 and ASSD are in pixels on the native grid and are undefined when either
mask is empty. `legacy_macro_metrics` additionally computes the class-macro
average over {background, lesion}, which is what the DeepLabv3+ benchmark used;
note that for a two-class confusion matrix macro-FPR equals 1 - macro-recall.

Usage
-----
    python3 analysis/code/seg_metrics_engine.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

REPO = Path(__file__).resolve().parents[2]
OUT_RESULTS = REPO / "analysis" / "results"
OUT_RESULTS.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20260904

# cohort -> (image_dir, ground_truth_mask_dir)
COHORTS = {
    "train": ("data/train/imagesTr", "data/train/labelsTr"),
    "internal_val": ("data/val/img_v", "data/val/seg_v"),
    "external_test1": ("data/test1/img_test1", "data/test1/seg_test1"),
    "external_test2": ("data/test2/img_test2", "data/test2/seg_test2"),
}

# cohort -> directory holding the nnU-Net predicted masks
NNUNET_PRED = {
    "train": "data/train/imagesTr_model",
    "internal_val": "data/val/img_v_model",
    "external_test1": "data/test1/img_test1_model",
    "external_test2": "data/test2/img_test2_model",
}


# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------
def case_id_of(path: Path) -> str:
    """Normalise any filename to case_XXXXX."""
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
    if digits:
        return f"case_{int(digits):05d}"
    return f"case_{base}"


def find_mask(case: str, mask_dir: Path) -> Path | None:
    """Resolve the mask file for a case id, handling both naming schemes."""
    num = int("".join(ch for ch in case if ch.isdigit()))
    for cand in (
        mask_dir / f"{case}.nii.gz",
        mask_dir / f"{case}.nii",
        mask_dir / f"{num}_seg.nii.gz",
        mask_dir / f"{num}_seg.nii",
        mask_dir / f"{num}.nii.gz",
        mask_dir / f"{num}.nii",
        mask_dir / f"{num:05d}_seg.nii.gz",
    ):
        if cand.exists():
            return cand
    return None


def load_binary_mask(path: Path) -> np.ndarray:
    """Load a NIfTI mask as a 2D binary array."""
    arr = np.squeeze(np.asarray(nib.load(str(path)).get_fdata()))
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 3:
        # single-slice volume stored with a trailing/leading singleton
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(f"unexpected 3D mask {path.name}: {arr.shape}")
    if arr.max() > 1.5:
        arr = arr / 255.0
    return (arr > 0.5).astype(np.uint8)


def align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Transpose ground truth if it is the transpose of the prediction."""
    if pred.shape == gt.shape:
        return gt
    if pred.shape == gt.T.shape:
        return gt.T
    raise ValueError(f"irreconcilable shapes pred={pred.shape} gt={gt.shape}")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
@dataclass
class CaseMetrics:
    case_id: str
    tp: int
    fp: int
    fn: int
    tn: int
    n_pixels: int
    gt_area: int
    pred_area: int
    gt_area_frac: float
    dice: float
    iou: float
    precision: float
    recall: float
    fpr: float
    hd95: float
    assd: float
    pred_empty: bool
    gt_empty: bool


def _boundary(mask: np.ndarray) -> np.ndarray:
    """Inner boundary pixels of a binary mask."""
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    eroded = ndimage.binary_erosion(mask.astype(bool), border_value=0)
    return mask.astype(bool) & ~eroded


def _surface_distances(pred: np.ndarray, gt: np.ndarray) -> np.ndarray | None:
    """
    Symmetric boundary distance set, in pixels.

    Returns None when either mask is empty (distance undefined).
    """
    if pred.sum() == 0 or gt.sum() == 0:
        return None
    bp, bg = _boundary(pred), _boundary(gt)
    if bp.sum() == 0 or bg.sum() == 0:
        return None
    # distance to nearest gt boundary pixel, evaluated at pred boundary pixels
    dt_to_gt = ndimage.distance_transform_edt(~bg)
    dt_to_pred = ndimage.distance_transform_edt(~bp)
    d_pred_to_gt = dt_to_gt[bp]
    d_gt_to_pred = dt_to_pred[bg]
    return np.concatenate([d_pred_to_gt, d_gt_to_pred])


def compute_case(case: str, pred: np.ndarray, gt: np.ndarray) -> CaseMetrics:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    tn = int(np.sum(~pred & ~gt))
    n = tp + fp + fn + tn

    def _safe(num, den):
        return float(num) / float(den) if den > 0 else float("nan")

    dice = _safe(2 * tp, 2 * tp + fp + fn)
    iou = _safe(tp, tp + fp + fn)
    precision = _safe(tp, tp + fp)
    recall = _safe(tp, tp + fn)
    fpr = _safe(fp, fp + tn)

    sd = _surface_distances(pred, gt)
    if sd is None:
        hd95 = float("nan")
        assd = float("nan")
    else:
        hd95 = float(np.percentile(sd, 95))
        assd = float(np.mean(sd))

    return CaseMetrics(
        case_id=case, tp=tp, fp=fp, fn=fn, tn=tn, n_pixels=n,
        gt_area=tp + fn, pred_area=tp + fp,
        gt_area_frac=_safe(tp + fn, n),
        dice=dice, iou=iou, precision=precision, recall=recall, fpr=fpr,
        hd95=hd95, assd=assd,
        pred_empty=bool(pred.sum() == 0), gt_empty=bool(gt.sum() == 0),
    )


def legacy_macro_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Class-macro-averaged metrics over {background, lesion}, as used by the
    DeepLabv3+ benchmark. Kept for comparison with the foreground-only values."""
    hist = np.zeros((2, 2), dtype=np.float64)
    np.add.at(hist, (gt.reshape(-1).astype(np.int64), pred.reshape(-1).astype(np.int64)), 1)
    tp = np.diag(hist)
    fp = hist.sum(axis=0) - tp
    fn = hist.sum(axis=1) - tp
    tn = hist.sum() - (tp + fp + fn)
    eps = 1e-8
    iu = tp / np.maximum(tp + fp + fn, eps)
    return {
        "macro_iou": float(np.nanmean(iu)),
        "macro_dice": float(np.nanmean((2 * tp) / np.maximum(2 * tp + fp + fn, eps))),
        "macro_precision": float(np.nanmean(tp / np.maximum(tp + fp, eps))),
        "macro_recall": float(np.nanmean(tp / np.maximum(tp + fn, eps))),
        "macro_fpr": float(np.nanmean(fp / np.maximum(fp + tn, eps))),
    }


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def bootstrap_ci(values: np.ndarray, n_boot: int = BOOTSTRAP_N,
                 seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float]:
    """Patient-level non-parametric bootstrap percentile CI of the mean."""
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(vals.mean())
    if vals.size < 2:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    boots = vals[idx].mean(axis=1)
    return mean, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def normal_ci(values: np.ndarray) -> tuple[float, float, float]:
    """Normal-approximation CI of the mean, as used by the original benchmarks."""
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(vals.mean())
    if vals.size < 2:
        return mean, mean, mean
    se = float(vals.std(ddof=1) / np.sqrt(vals.size))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def summarise(cases: list[CaseMetrics]) -> dict:
    out: dict = {"n_cases": len(cases)}
    for field in ("dice", "iou", "precision", "recall", "fpr", "hd95", "assd"):
        vals = np.array([getattr(c, field) for c in cases], dtype=np.float64)
        m, lo, hi = bootstrap_ci(vals)
        nm, nlo, nhi = normal_ci(vals)
        out[field] = {
            "mean": m, "boot_lo": lo, "boot_hi": hi,
            "normal_lo": nlo, "normal_hi": nhi,
            "n_valid": int(np.isfinite(vals).sum()),
        }
    out["n_pred_empty"] = int(sum(c.pred_empty for c in cases))
    out["n_gt_empty"] = int(sum(c.gt_empty for c in cases))
    out["mean_gt_area_frac"] = float(np.mean([c.gt_area_frac for c in cases]))
    return out


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def evaluate_model(model_name: str, pred_dirs: dict[str, str],
                   verbose: bool = True) -> dict:
    """Evaluate one model across all cohorts from its predicted masks."""
    all_results: dict = {}
    per_case_rows: list[dict] = []

    for cohort, (img_rel, gt_rel) in COHORTS.items():
        if cohort not in pred_dirs:
            continue
        gt_dir = REPO / gt_rel
        pred_dir = REPO / pred_dirs[cohort]
        img_dir = REPO / img_rel

        cases = sorted({case_id_of(p) for p in img_dir.glob("*.nii.gz")})
        if not cases:
            # The cohort images are not redistributable, so a clone of this
            # repository has nothing to score. Skip rather than emit NaN rows.
            if verbose:
                print(f"  {model_name:12s} {cohort:15s} no images under "
                      f"{img_rel}, skipping")
            continue

        metrics: list[CaseMetrics] = []
        macro_acc: list[dict] = []
        missing = []

        for case in cases:
            gt_path = find_mask(case, gt_dir)
            pred_path = find_mask(case, pred_dir)
            if gt_path is None or pred_path is None:
                missing.append(case)
                continue
            gt = load_binary_mask(gt_path)
            pred = load_binary_mask(pred_path)
            try:
                gt = align(pred, gt)
            except ValueError as e:
                missing.append(f"{case}({e})")
                continue
            metrics.append(compute_case(case, pred, gt))
            macro_acc.append(legacy_macro_metrics(pred, gt))

        summary = summarise(metrics)
        summary["n_missing"] = len(missing)
        summary["missing"] = missing[:20]
        summary["legacy_macro"] = {
            k: float(np.mean([m[k] for m in macro_acc])) for k in macro_acc[0]
        } if macro_acc else {}
        all_results[cohort] = summary

        for c in metrics:
            row = asdict(c)
            row["model"] = model_name
            row["cohort"] = cohort
            per_case_rows.append(row)

        if verbose:
            d = summary["dice"]
            f = summary["fpr"]
            print(f"  {model_name:12s} {cohort:15s} n={summary['n_cases']:4d} "
                  f"Dice={d['mean']:.4f} [{d['boot_lo']:.4f}-{d['boot_hi']:.4f}]  "
                  f"FPR={f['mean']:.5f}  HD95={summary['hd95']['mean']:.2f}px  "
                  f"empty_pred={summary['n_pred_empty']} missing={len(missing)}")

    return {"summary": all_results, "per_case": per_case_rows}


def main() -> None:
    import csv

    print("=" * 78)
    print("SEGMENTATION METRIC ENGINE - recomputation from locked masks")
    print("=" * 78)

    results = {}
    nnunet_dirs = {
        cohort: rel for cohort, rel in NNUNET_PRED.items()
        if (REPO / rel).exists() and any((REPO / rel).glob("*.nii.gz"))
    }
    if nnunet_dirs:
        print("\nnnU-Net (locked predicted masks in data/*/*_model):")
        scored = evaluate_model("nnUNet", nnunet_dirs)
        if scored["summary"]:
            results["nnUNet"] = scored
    else:
        print("\nnnU-Net: predicted masks not found under data/*/*_model, skipping "
              "(the cohort images are not redistributable, see data/README.md)")

    # The other two models are evaluated once run_seg_inference.py has written
    # their masks into analysis/masks_locked/.
    locked_root = REPO / "analysis" / "masks_locked"
    for model_name in ("DeepLabv3plus", "UNet"):
        dirs = {}
        for cohort in COHORTS:
            d = locked_root / model_name / cohort
            if d.exists() and any(d.glob("*.nii.gz")):
                dirs[cohort] = str(d.relative_to(REPO))
        if dirs:
            print(f"\n{model_name} (locked predicted masks):")
            scored = evaluate_model(model_name, dirs)
            if scored["summary"]:
                results[model_name] = scored
        else:
            print(f"\n{model_name}: locked masks not found, skipping "
                  f"(run run_seg_inference.py first)")

    if not results:
        print("\nNo predicted masks were found for any model, so nothing was "
              "recomputed and the deposited results were left in place.")
        return

    with (OUT_RESULTS / "seg_metrics_summary.json").open("w") as f:
        json.dump({k: v["summary"] for k, v in results.items()}, f, indent=2)

    rows = [r for v in results.values() for r in v["per_case"]]
    if rows:
        with (OUT_RESULTS / "seg_metrics_per_case.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"\nWrote {OUT_RESULTS/'seg_metrics_summary.json'}")
    print(f"Wrote {OUT_RESULTS/'seg_metrics_per_case.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
