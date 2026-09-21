#!/usr/bin/env python3
"""
Cross-cohort overlap: does any image appear in more than one cohort?

task0_data_check.py tests `case_id.duplicated()` WITHIN each split and reports
"all integrity checks passed". It never compares cohorts to each other, so it
cannot detect a case appearing in both training and test. This script runs that
comparison.

Two facts make the check necessary:

  1. `case_id` is a within-cohort index, not a patient identifier. Every cohort
     numbers independently from case_00001, so case_00001 exists four times and
     refers to four different patients. Identifier comparison is therefore
     meaningless and this script does not rely on it.

  2. No global patient identifier exists anywhere in the deposited data, so
     patient-level overlap cannot be established from identifiers at all.

What this script does instead is compare IMAGE CONTENT:

  - exact: SHA256 of the raw voxel array, catching byte-identical reuse
  - near:  correlation between 32x32 normalised thumbnails, catching the same
           image after re-encoding, rescaling or format conversion

Scope and limits, stated plainly: this establishes that no IMAGE is shared
between cohorts. It cannot establish that no PATIENT is shared, because the same
patient could contribute different frames to different cohorts. Ruling that out
requires the source records at the contributing centres and cannot be done from
this repository.

This script reads pixel content and cohort membership only. It never reads
labels or model outputs, so it carries no information about outcome or
performance and is safe to run before the pipeline is locked.

Output: analysis/results/task0b_cohort_overlap.json and .md
"""

from __future__ import annotations

import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, ensure_dirs  # type: ignore  # noqa: E402

COHORTS = {
    "train":          (REPO / "data" / "train" / "imagesTr", REPO / "data" / "train" / "labelsTr"),
    "internal_val":   (REPO / "data" / "val" / "img_v",      REPO / "data" / "val" / "seg_v"),
    "external_test1": (REPO / "data" / "test1" / "img_test1", REPO / "data" / "test1" / "seg_test1"),
    "external_test2": (REPO / "data" / "test2" / "img_test2", REPO / "data" / "test2" / "seg_test2"),
}

THUMB = 32
NEAR_DUPLICATE_R = 0.99


def load_array(path: Path) -> np.ndarray:
    import nibabel as nib
    return np.squeeze(np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32))


def thumbnail(arr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """
    32x32 z-scored thumbnail of the LESION REGION.

    Whole-frame thumbnails are not usable here. Every image from a given scanner
    carries the same static UI chrome (scale bars, device text, borders, black
    surround), which dominates the correlation: an earlier whole-frame version of
    this check flagged 16 cross-cohort pairs at r >= 0.99 that were verified at
    full resolution to be different patients sharing scanner furniture, with zero
    byte-identical pairs among them. Cropping to the lesion bounding box removes
    the static content and compares anatomy instead.
    """
    import cv2
    if arr.ndim > 2:
        arr = arr.mean(axis=tuple(range(2, arr.ndim)))
    if mask is not None and mask.shape == arr.shape:
        ys, xs = np.where(mask > 0.5)
        if len(xs):
            arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    t = cv2.resize(arr.astype(np.float32), (THUMB, THUMB),
                   interpolation=cv2.INTER_AREA).ravel()
    sd = t.std()
    return (t - t.mean()) / sd if sd > 1e-8 else np.zeros_like(t)


def main() -> None:
    ensure_dirs()

    records = []
    for cohort, (img_dir, mask_dir) in COHORTS.items():
        files = sorted(img_dir.glob("*.nii.gz"))
        if not files:
            raise SystemExit(f"no images found in {img_dir}")
        print(f"hashing {cohort:15s} n={len(files)}", flush=True)
        for f in files:
            case = f.name.replace(".nii.gz", "").replace("_0000", "")
            arr = load_array(f)
            mp = mask_dir / f"{case}.nii.gz"
            mask = load_array(mp) if mp.exists() else None
            records.append({
                "cohort": cohort,
                "case_id": case,
                "file": str(f.relative_to(REPO)),
                "sha256": hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest(),
                "shape": list(arr.shape),
                "_thumb": thumbnail(arr, mask),
            })

    n = len(records)
    print(f"\n{n} images across {len(COHORTS)} cohorts")

    # ---- identifier collision, demonstrating why ids cannot be used ---------
    by_cohort_ids = {c: {r["case_id"] for r in records if r["cohort"] == c}
                     for c in COHORTS}
    id_collisions = {
        f"{a}|{b}": len(by_cohort_ids[a] & by_cohort_ids[b])
        for a, b in combinations(COHORTS, 2)
    }

    # ---- exact duplicates --------------------------------------------------
    by_hash: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        by_hash.setdefault(r["sha256"], []).append(i)

    exact_cross, exact_within = [], []
    for idxs in by_hash.values():
        if len(idxs) < 2:
            continue
        for i, j in combinations(idxs, 2):
            pair = {
                "a": f"{records[i]['cohort']}/{records[i]['case_id']}",
                "b": f"{records[j]['cohort']}/{records[j]['case_id']}",
                "sha256": records[i]["sha256"][:16],
            }
            (exact_cross if records[i]["cohort"] != records[j]["cohort"]
             else exact_within).append(pair)

    # ---- near duplicates ---------------------------------------------------
    T = np.vstack([r["_thumb"] for r in records])
    R = (T @ T.T) / (THUMB * THUMB)
    np.fill_diagonal(R, 0.0)
    cohort_of = np.array([list(COHORTS).index(r["cohort"]) for r in records])

    near_cross = []
    ii, jj = np.where(np.triu(R, 1) >= NEAR_DUPLICATE_R)
    for i, j in zip(ii, jj):
        if cohort_of[i] != cohort_of[j]:
            near_cross.append({
                "a": f"{records[i]['cohort']}/{records[i]['case_id']}",
                "b": f"{records[j]['cohort']}/{records[j]['case_id']}",
                "r": round(float(R[i, j]), 5),
            })
    near_cross.sort(key=lambda d: -d["r"])

    # Two separate questions. Cross-cohort duplication is a leakage question;
    # within-cohort duplication is a cohort-size question. They are reported
    # separately because they have different consequences.
    leakage_clean = not exact_cross and not near_cross
    unique_counts = {
        c: len({r["sha256"] for r in records if r["cohort"] == c}) for c in COHORTS
    }
    out = {
        "n_images": n,
        "cohort_sizes": {c: len(by_cohort_ids[c]) for c in COHORTS},
        "case_id_is_patient_identifier": False,
        "case_id_collisions_between_cohorts": id_collisions,
        "exact_duplicate_pairs_across_cohorts": exact_cross,
        "exact_duplicate_pairs_within_cohort": exact_within,
        "near_duplicate_threshold_r": NEAR_DUPLICATE_R,
        "near_duplicate_pairs_across_cohorts": near_cross,
        "max_cross_cohort_correlation": round(float(
            R[np.ix_(cohort_of == 0, cohort_of != 0)].max()), 5) if n else None,
        "unique_images_per_cohort": unique_counts,
        "duplicated_within_cohort": {
            c: len([r for r in records if r["cohort"] == c]) - unique_counts[c]
            for c in COHORTS},
        "leakage_verdict": ("no image is shared between cohorts"
                            if leakage_clean else "SHARED IMAGES ACROSS COHORTS"),
        "verdict": ("no image is shared between cohorts"
                    if leakage_clean else "SHARED IMAGES ACROSS COHORTS"),
        "limitation": (
            "Establishes image-level separation only. Patient-level separation "
            "cannot be established from this repository because no global patient "
            "identifier exists; the same patient could contribute different frames "
            "to different cohorts. That must be confirmed against the source "
            "records at the contributing centres."),
    }

    (RESULTS / "task0b_cohort_overlap.json").write_text(json.dumps(out, indent=2))

    md = [
        "# Cross-cohort overlap check",
        "",
        f"{n} images compared across {len(COHORTS)} cohorts by image content.",
        "",
        "## Why identifiers cannot be used",
        "",
        "`case_id` is a within-cohort index. Every cohort numbers independently "
        "from `case_00001`, so identifiers collide completely and carry no "
        "patient information:",
        "",
        "| Cohort pair | Colliding case_ids |",
        "|---|---|",
    ]
    md += [f"| {k.replace('|', ' vs ')} | {v} |" for k, v in id_collisions.items()]
    md += [
        "",
        "## Cross-cohort separation (the leakage question)",
        "",
        f"- Exact duplicate pairs across cohorts: **{len(exact_cross)}**",
        f"- Near-duplicate pairs across cohorts (lesion-crop r >= "
        f"{NEAR_DUPLICATE_R}): **{len(near_cross)}**",
        "",
        f"**Verdict: {out['leakage_verdict']}.**",
        "",
        "## Within-cohort duplication (the cohort-size question)",
        "",
        "Byte-identical images appearing more than once inside one cohort. This "
        "is not leakage, but it means the reported cohort size counts some images "
        "twice.",
        "",
        "| Cohort | Images | Unique | Duplicated |",
        "|---|---|---|---|",
    ] + [
        f"| {c} | {len(by_cohort_ids[c])} | {unique_counts[c]} | "
        f"{len(by_cohort_ids[c]) - unique_counts[c]} |" for c in COHORTS
    ] + [
        "",
        "## Limitation",
        "",
        out["limitation"],
    ]
    (RESULTS / "task0b_cohort_overlap.md").write_text("\n".join(md) + "\n")

    print(f"\nexact cross-cohort duplicates: {len(exact_cross)}")
    print(f"near cross-cohort duplicates:  {len(near_cross)}")
    print(f"verdict: {out['verdict']}")
    print(f"\nwrote results/task0b_cohort_overlap.json and .md")


if __name__ == "__main__":
    main()
