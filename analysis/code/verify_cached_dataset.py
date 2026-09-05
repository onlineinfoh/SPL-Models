"""
Check that CachedTightCropDataset returns the same tensors as train.py's
TightCropDataset, in both evaluation and training (augmented) mode.

The augmented comparison reseeds numpy before each __getitem__ so both
implementations draw the same augmentation parameters. Exits non-zero on any
mismatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "binary_classification"))
sys.path.insert(0, str(REPO / "analysis" / "code"))

from train import (  # type: ignore  # noqa: E402
    INT_IMG_DIR, INT_MASK_DIR, LABEL_FILES, TRAIN_IMG_DIR, TRAIN_MASK_DIR,
    TightCropDataset, load_labels, normalize_case_prefix,
)
from cached_dataset import CachedTightCropDataset  # type: ignore  # noqa: E402


def compare(orig, cached, indices, is_train: bool, tag: str) -> bool:
    ok = True
    for i in indices:
        np.random.seed(1000 + i)
        xo, yo, mo, co = orig[i]
        np.random.seed(1000 + i)
        xc, yc, mc, cc = cached[i]

        same_x = torch.equal(xo, xc)
        same_m = torch.equal(mo, mc)
        same_y = torch.equal(yo, yc)
        same_c = (co == cc)
        if not (same_x and same_m and same_y and same_c):
            ok = False
            dmax = (xo - xc).abs().max().item()
            print(f"  MISMATCH {tag} idx={i} case={co}/{cc} "
                  f"x={same_x} (maxdiff={dmax:.3e}) mask={same_m} y={same_y}")
    print(f"  {tag}: {'IDENTICAL' if ok else 'DIFFERS'} "
          f"over {len(indices)} samples (is_train={is_train})")
    return ok


def main() -> None:
    lbl_train_all = load_labels(LABEL_FILES["train"])
    train_map = {
        normalize_case_prefix(p.name): lbl_train_all[normalize_case_prefix(p.name)]
        for p in TRAIN_IMG_DIR.glob("*.nii.gz")
        if normalize_case_prefix(p.name) in lbl_train_all
    }
    lbl_int = load_labels(LABEL_FILES["internal"])

    all_ok = True
    print("Verifying cached dataset equivalence")

    for is_train, img_d, mask_d, lbls, tag in [
        (False, INT_IMG_DIR, INT_MASK_DIR, lbl_int, "internal_val eval-mode"),
        (False, TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_map, "train eval-mode"),
        (True, TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_map, "train augmented-mode"),
    ]:
        orig = TightCropDataset(img_d, mask_d, lbls, is_train=is_train)
        cached = CachedTightCropDataset(img_d, mask_d, lbls, is_train=is_train)
        assert len(orig) == len(cached), f"length mismatch {len(orig)} vs {len(cached)}"
        n = len(orig)
        idx = sorted(set(list(range(0, min(40, n))) +
                         list(np.linspace(0, n - 1, 40).astype(int))))
        all_ok &= compare(orig, cached, idx, is_train, tag)

    print()
    if all_ok:
        print("PASS: cached dataset is bit-identical to the original.")
    else:
        print("FAIL: cached dataset differs. Do not use for the sweep.")
        sys.exit(1)


if __name__ == "__main__":
    main()
