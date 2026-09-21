#!/usr/bin/env python3
# ==========================================================================
# SUPERSEDED BY THE LOCKED RE-ANALYSIS
# ==========================================================================
#
# Frozen-pipeline evaluation of DenseNet121, written during the re-analysis and
# replaced by analysis/code/export_locked_predictions.py.
#
# Retained unmodified as the audit record. Produces no reported result.
# See README.md and docs/REPRODUCE.md for the active pipeline.
# ==========================================================================
#
"""
Run the frozen two-stage pipeline over all four cohorts.

Stage 1  nnU-Net 2d, fold_all, checkpoint_best.pth  (masks already predicted,
         present as the *_model sibling directories)
Stage 2  DenseNet121, runs/densenet121/best.pth     (the published model)

Nothing is trained, selected or tuned here. The models are loaded as they stand
and applied once per cohort. Both mask variants are scored:

    gt     manual reference masks      -> upper bound, mask quality removed
    model  nnU-Net predicted masks     -> the deployable pipeline

Both input resolutions are scored, which settles the 300/224 discrepancy
empirically rather than by assertion: training used 300 and the published
inference path used 224.

Output
------
analysis/results/frozen_pipeline_probs_<cohort>_<variant>_<size>.csv
    case_id, cohort_case_id, label, prob_malignant
analysis/results/frozen_pipeline_metrics.csv
analysis/results/frozen_pipeline_metrics.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "binary_classification"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import infer_probs_tight as ipt  # type: ignore  # noqa: E402
from train import build_model  # type: ignore  # noqa: E402
from common import (RESULTS, ensure_dirs, delong_auc_ci, threshold_metrics,  # noqa: E402
                    threshold_max_accuracy, threshold_youden)

ARCH = "densenet121"
CKPT = REPO / "binary_classification" / "runs" / ARCH / "best.pth"
SIZES = [224, 300]

PREFIX = {"train": "train", "internal_val": "ival",
          "external_test1": "ext1", "external_test2": "ext2"}

COHORTS = {
    "train":          ("train", "imagesTr", "labelsTr",  "labels_train.csv"),
    "internal_val":   ("val",   "img_v",    "seg_v",     "labels_internal_val.csv"),
    "external_test1": ("test1", "img_test1", "seg_test1", "labels_external_test1.csv"),
    "external_test2": ("test2", "img_test2", "seg_test2", "labels_external_test2.csv"),
}


def score(model, img_dir: Path, mask_dir: Path, labels: dict, device):
    ids, ys, ps = [], [], []
    for ip in sorted(img_dir.glob("*.nii.gz")):
        case = ipt.normalize_case_prefix(ip.name)
        if case not in labels:
            continue
        try:
            x = ipt.load_image_mask(ip, mask_dir / f"{case}.nii.gz", mask_dir)
        except FileNotFoundError:
            continue
        with torch.no_grad():
            ps.append(float(torch.sigmoid(model(x[None].to(device))).item()))
        ys.append(int(labels[case]))
        ids.append(case)
    return ids, np.asarray(ys, int), np.asarray(ps, float)


def main() -> None:
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(ARCH).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()
    print(f"frozen model: {CKPT.relative_to(REPO)}\n")

    rows = []
    thresholds: dict[tuple[int, str], dict] = {}

    for size in SIZES:
        ipt.IMG_SIZE = size
        for cohort, (sub, imgd, maskd, labf) in COHORTS.items():
            labels = ipt.load_labels(REPO / "binary_classification" / "labels" / labf)
            for variant in ("gt", "model"):
                img_dir = REPO / "data" / sub / imgd
                mask_dir = (REPO / "data" / sub / maskd if variant == "gt"
                            else REPO / "data" / sub / f"{imgd}_model")
                if not mask_dir.exists():
                    print(f"  skip {cohort}/{variant}: no {mask_dir.name}")
                    continue

                ids, y, p = score(model, img_dir, mask_dir, labels, device)

                out = RESULTS / f"frozen_pipeline_probs_{cohort}_{variant}_{size}.csv"
                with open(out, "w") as fh:
                    fh.write("case_id,cohort_case_id,label,prob_malignant\n")
                    for c, yi, pi in zip(ids, y, p):
                        fh.write(f"{PREFIX[cohort]}_{c},{c},{int(yi)},{pi:.6f}\n")

                # Operating point is derived on internal_val ONLY and carried to
                # every other cohort unchanged. Deriving it per cohort would make
                # the external numbers optimistic.
                if cohort == "internal_val":
                    thresholds[(size, variant)] = {
                        "max_acc": threshold_max_accuracy(y, p),
                        "youden": threshold_youden(y, p),
                    }
                thr = thresholds.get((size, variant), {}).get("youden", 0.5)

                m = threshold_metrics(y, p, thr)
                auc = delong_auc_ci(y, p) if len(np.unique(y)) > 1 else {}
                rows.append({
                    "cohort": cohort, "mask_variant": variant, "img_size": size,
                    "n": len(y), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
                    "threshold_from_internal_val": round(thr, 6),
                    "auc": round(auc.get("auc", float("nan")), 4),
                    "auc_lo": round(auc.get("delong_logit_lo", float("nan")), 4),
                    "auc_hi": round(auc.get("delong_logit_hi", float("nan")), 4),
                    "accuracy": round(m["accuracy"], 4),
                    "sensitivity": round(m["sensitivity"], 4),
                    "specificity": round(m["specificity"], 4),
                    "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
                })
                print(f"  {size}px {cohort:15s} {variant:5s} n={len(y):4d} "
                      f"auc={rows[-1]['auc']:.4f} thr={thr:.4f}")

    hdr = list(rows[0])
    with open(RESULTS / "frozen_pipeline_metrics.csv", "w") as fh:
        fh.write(",".join(hdr) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in hdr) + "\n")

    md = ["# Frozen pipeline evaluation", "",
          f"Stage 1 nnU-Net 2d fold_all (predicted masks). "
          f"Stage 2 {ARCH} `runs/{ARCH}/best.pth`.", "",
          "Nothing trained or selected here. The operating point is derived on "
          "internal_val by Youden and applied unchanged to every other cohort.", "",
          "| Cohort | Mask | Size | n | AUC (95% CI) | Sens | Spec | Acc |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['cohort']} | {r['mask_variant']} | {r['img_size']} | "
                  f"{r['n']} | {r['auc']:.3f} ({r['auc_lo']:.3f}-{r['auc_hi']:.3f}) | "
                  f"{r['sensitivity']:.3f} | {r['specificity']:.3f} | {r['accuracy']:.3f} |")
    (RESULTS / "frozen_pipeline_metrics.md").write_text("\n".join(md) + "\n")

    print(f"\nwrote results/frozen_pipeline_metrics.csv and .md")


if __name__ == "__main__":
    main()
