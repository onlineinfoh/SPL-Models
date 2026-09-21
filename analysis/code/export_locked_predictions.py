#!/usr/bin/env python3
"""
Score the locked model on all four cohorts and both mask variants.

Writes per-case probability files in the schema the analysis layer already
consumes, so every downstream statistic can be regenerated on the locked model
without duplicating any statistical code:

    <out>/<arch>/{split}_seed{seed}_probs.txt          manual masks  ("gt")
    <out>/<arch>/{split}_model_seed{seed}_probs.txt    predicted     ("model")

Preprocessing comes from protocol/run_protocol.py::LockedDataset, so these
probabilities are produced by exactly the pipeline the lock record describes:
300x300 at both training and inference, 10 percent halo, rotation defect
corrected. They are NOT produced by binary_classification/infer_probs_tight.py,
which is the superseded 224 px path.

The model that scores here, the seed, and the threshold all come from
protocol/locked_pipeline.json. Nothing is selected or tuned in this script.

Hold-out note: this runs after Phase D, so the external cohorts are legitimately
readable. The gate is deliberately not installed; the lock record already exists
and `protocol/gate_audit.jsonl` records the release that preceded this step.

Usage
-----
    python analysis/code/export_locked_predictions.py
    python analysis/code/export_locked_predictions.py --variants gt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "binary_classification"))
sys.path.insert(0, str(REPO / "analysis" / "code"))

from protocol.run_protocol import locked_loader, predict  # noqa: E402
from train import (  # type: ignore  # noqa: E402
    LABEL_FILES, TRAIN_IMG_DIR, TRAIN_MASK_DIR, INT_IMG_DIR, INT_MASK_DIR,
    EXT1_IMG_DIR, EXT1_MASK_DIR, EXT2_IMG_DIR, EXT2_MASK_DIR,
    build_model, load_labels, normalize_case_prefix,
)

LOCK = REPO / "protocol" / "locked_pipeline.json"
DEFAULT_OUT = REPO / "protocol" / "results" / "predictions_locked"

# split -> (images, manual masks, label file key)
COHORTS = {
    "train":          (TRAIN_IMG_DIR, TRAIN_MASK_DIR, "train"),
    "internal_val":   (INT_IMG_DIR, INT_MASK_DIR, "internal"),
    "external_test1": (EXT1_IMG_DIR, EXT1_MASK_DIR, "external1"),
    "external_test2": (EXT2_IMG_DIR, EXT2_MASK_DIR, "external2"),
}


_MASK_ROOT = None   # set from --nnunet-masks


def predicted_mask_dir(img_dir: Path, split: str = "") -> Path:
    """Directory of nnU-Net predicted masks for these images."""
    if _MASK_ROOT is not None:
        return _MASK_ROOT / split
    return img_dir.parent / f"{img_dir.name}_model"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--variants", nargs="+", default=["gt", "model"],
                    choices=["gt", "model"])
    ap.add_argument("--seed", type=int, default=None,
                    help="Override the locked seed. Used ONLY for the declared "
                         "robustness repeats (1234, 2025), which are scored "
                         "after the lock to test whether the result is stable "
                         "across seeds. This is a robustness analysis, not a "
                         "selection step: the architecture and threshold remain "
                         "those in locked_pipeline.json.")
    ap.add_argument("--nnunet-masks", type=Path, default=None,
                    help="Root with one subdirectory per cohort holding nnU-Net "
                         "predicted masks, e.g. predicted_masks_v2. Defaults to "
                         "the published *_model siblings.")
    args = ap.parse_args()

    if not LOCK.exists():
        raise SystemExit(f"no lock record at {LOCK}; run Phase D first")
    lock = json.loads(LOCK.read_text())
    arch = lock["selected_architecture"]
    seed = args.seed if args.seed is not None else lock["selected_seed"]
    if args.seed is not None and args.seed != lock["selected_seed"]:
        print(f"!! robustness repeat: seed {args.seed} (locked seed is "
              f"{lock['selected_seed']}). Not a selection step.\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = REPO / "protocol" / "runs" / f"{arch}_seed{seed}" / "best.pth"
    if not ckpt.exists():
        raise SystemExit(f"checkpoint missing: {ckpt}")

    model = build_model(arch).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    global _MASK_ROOT
    _MASK_ROOT = args.nnunet_masks
    out_dir = args.out / arch
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"locked model : {arch} seed {seed} epoch {lock['selected_epoch']}")
    print(f"threshold    : {lock['threshold']:.4f} ({lock['threshold_rule']})")
    print(f"output       : {out_dir.relative_to(REPO)}\n")

    manifest = []
    for split, (img_dir, gt_mask_dir, lbl_key) in COHORTS.items():
        labels = load_labels(LABEL_FILES[lbl_key])
        if split == "train":
            labels = {normalize_case_prefix(p.name): labels[normalize_case_prefix(p.name)]
                      for p in img_dir.glob("*.nii.gz")
                      if normalize_case_prefix(p.name) in labels}

        for variant in args.variants:
            mask_dir = gt_mask_dir if variant == "gt" else predicted_mask_dir(img_dir, split)
            if not mask_dir.exists():
                print(f"  SKIP {split:15s} {variant:5s}  no {mask_dir.name}")
                continue

            loader = locked_loader(img_dir, mask_dir, labels, 16, False)
            ids, y, p = predict(model, loader, device)

            suffix = "" if variant == "gt" else "_model"
            path = out_dir / f"{split}{suffix}_seed{seed}_probs.txt"
            with open(path, "w") as fh:
                fh.write("case_id,label,prob_malignant,prob_benign\n")
                for cid, yi, pi in zip(ids, y, p):
                    fh.write(f"{cid},{int(yi)},{pi:.6f},{1.0 - pi:.6f}\n")

            n_pos, n_neg = int(y.sum()), int((1 - y).sum())
            print(f"  {split:15s} {variant:5s} n={len(y):4d} "
                  f"({n_pos} malignant / {n_neg} benign) -> {path.name}")
            manifest.append({
                "split": split, "variant": variant, "n": len(y),
                "n_pos": n_pos, "n_neg": n_neg,
                "file": str(path.relative_to(REPO)),
                "prob_min": round(float(p.min()), 6),
                "prob_max": round(float(p.max()), 6),
            })

    (args.out / "manifest.json").write_text(json.dumps({
        "source": "protocol/locked_pipeline.json",
        "architecture": arch, "seed": seed,
        "threshold": lock["threshold"],
        "git_commit": lock["git_commit"],
        "preprocessing": "protocol/run_protocol.py::LockedDataset, 300x300",
        "files": manifest,
    }, indent=2))
    print(f"\nwrote {len(manifest)} files + manifest.json")


if __name__ == "__main__":
    main()
