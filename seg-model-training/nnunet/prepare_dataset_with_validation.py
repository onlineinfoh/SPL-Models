#!/usr/bin/env python3
"""
Build Dataset001_lungval: the segmentation dataset with a real validation split.

Why this exists
---------------
Dataset000_lung was trained with `-f all`. In nnU-Net v2, fold "all" sets
`val_keys = tr_keys` (nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:565-569),
so the run had no held-out validation: `fold_all/validation/` contains exactly
the 600 training cases, and the reported "Mean Validation Dice: 0.9825" is
training Dice. The selected segmentation model was therefore checkpointed on its
own training data - the same defect the reviewer identified in the U-Net
baseline, present in the model that was actually chosen.

The fix does not need a custom trainer. nnU-Net reads `splits_final.json` from
the preprocessed folder, so putting both cohorts in one dataset and defining
fold 0 explicitly gives a genuine held-out validation set using stock nnU-Net.

Case identifiers
----------------
`case_00001` exists in BOTH cohorts and refers to two different patients
(analysis/results/task0b_cohort_overlap.md). Internal-validation cases are
therefore renumbered into a disjoint range rather than copied verbatim, and the
mapping is written out so every renumbered case can be traced back.

Dataset000_lung is left untouched.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NNUNET = REPO / "seg-model-training" / "nnunet"
RAW = NNUNET / "nnUNet_raw"
SRC = RAW / "Dataset000_lung"
DST = RAW / "Dataset001_lungval"

TRAIN_IMG = REPO / "data" / "train" / "imagesTr"
TRAIN_MSK = REPO / "data" / "train" / "labelsTr"
VAL_IMG = REPO / "data" / "val" / "img_v"
VAL_MSK = REPO / "data" / "val" / "seg_v"

VAL_ID_OFFSET = 1000          # internal_val becomes case_01001 ...


def case_of(p: Path) -> str:
    n = p.name[:-7] if p.name.endswith(".nii.gz") else p.stem
    return n[:-5] if n.endswith("_0000") else n


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"missing source dataset {SRC}")
    if DST.exists():
        raise SystemExit(f"{DST} already exists; remove it to rebuild")

    (DST / "imagesTr").mkdir(parents=True)
    (DST / "labelsTr").mkdir(parents=True)

    training, mapping = [], []

    # ---- training cohort, identifiers preserved --------------------------
    src_imgs = sorted((SRC / "imagesTr").glob("*.nii.gz"))
    print(f"copying {len(src_imgs)} training cases")
    for ip in src_imgs:
        case = case_of(ip)
        lp = SRC / "labelsTr" / f"{case}.nii.gz"
        if not lp.exists():
            raise SystemExit(f"missing label for {case}")
        shutil.copy2(ip, DST / "imagesTr" / f"{case}_0000.nii.gz")
        shutil.copy2(lp, DST / "labelsTr" / f"{case}.nii.gz")
        training.append(case)
        mapping.append({"new_id": case, "cohort": "train", "original_id": case})

    # ---- internal validation cohort, renumbered --------------------------
    val_imgs = sorted(VAL_IMG.glob("*.nii.gz"))
    print(f"copying {len(val_imgs)} internal_val cases (renumbered +{VAL_ID_OFFSET})")
    validation = []
    for ip in val_imgs:
        orig = case_of(ip)
        num = int("".join(ch for ch in orig if ch.isdigit()))
        new = f"case_{num + VAL_ID_OFFSET:05d}"
        lp = VAL_MSK / f"{orig}.nii.gz"
        if not lp.exists():
            raise SystemExit(f"missing internal_val mask for {orig} at {lp}")
        shutil.copy2(ip, DST / "imagesTr" / f"{new}_0000.nii.gz")
        shutil.copy2(lp, DST / "labelsTr" / f"{new}.nii.gz")
        validation.append(new)
        mapping.append({"new_id": new, "cohort": "internal_val", "original_id": orig})

    assert not set(training) & set(validation), "identifier collision after renumbering"

    # ---- dataset.json ----------------------------------------------------
    src_meta = json.loads((SRC / "dataset.json").read_text())
    meta = {
        "name": "Dataset001_lungval",
        "description": (
            "Lung lesion segmentation. Training cohort (Center 1, n=600) plus "
            "internal validation cohort (Center 1, n=257) as a held-out split. "
            "Supersedes Dataset000_lung, which was trained with fold 'all' and "
            "therefore validated on its own training data."),
        "reference": src_meta.get("reference", "Internal collection"),
        "licence": src_meta.get("licence", "Proprietary, for research use only"),
        "release": "2.0",
        "channel_names": {"0": src_meta.get("modality", {}).get("0", "CT")},
        "labels": src_meta.get("labels", {"background": 0, "tumor": 1}),
        "numTraining": len(training) + len(validation),
        "file_ending": ".nii.gz",
        "training": [
            {"image": f"./imagesTr/{c}_0000.nii.gz", "label": f"./labelsTr/{c}.nii.gz"}
            for c in training + validation],
    }
    (DST / "dataset.json").write_text(json.dumps(meta, indent=2))
    (DST / "case_id_mapping.json").write_text(json.dumps({
        "note": ("case_id is a within-cohort index in the source data and collides "
                 "across cohorts; internal_val was renumbered by "
                 f"+{VAL_ID_OFFSET} to make identifiers disjoint."),
        "n_train": len(training), "n_internal_val": len(validation),
        "mapping": mapping,
    }, indent=2))

    # ---- splits_final.json ----------------------------------------------
    # Single fold. Training cohort trains; internal validation validates. This
    # is what gives checkpoint_best.pth a genuinely held-out selection signal.
    splits = [{"train": training, "val": validation}]
    NNUNET.joinpath("splits_final.json").write_text(json.dumps(splits, indent=2))

    print(f"\nDataset001_lungval: {len(training)} train / {len(validation)} val")
    print(f"  raw:    {DST.relative_to(REPO)}")
    print(f"  splits: seg-model-training/nnunet/splits_final.json")
    print("\nnext:")
    print("  nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity -c 2d")
    print("  cp seg-model-training/nnunet/splits_final.json \\")
    print("     seg-model-training/nnunet/nnUNet_preprocessed/Dataset001_lungval/")
    print("  nnUNetv2_train 1 2d 0")


if __name__ == "__main__":
    main()
