#!/usr/bin/env python3
"""
Post-hoc probability calibration for the locked model.

Why
---
The locked model discriminates well but its probabilities are badly scaled. On
the tuning cohort the calibration slope is 3.46 where 1.0 is ideal, meaning the
predicted probabilities are compressed toward the middle of the range. The cause
is the retained checkpoint being epoch 4 of 40: AUC-based early stopping found a
model that ranks cases well before its logits had spread out, so discrimination
and calibration decoupled.

This matters for two reported analyses. Decision-curve analysis assumes
calibrated probabilities, and a reported "malignancy probability" that is not a
probability cannot support clinical interpretation.

Method
------
Platt scaling: fit y ~ a + b * logit(p) on the tuning cohort, then map every
cohort through sigmoid(a + b * logit(p)).

Platt rather than isotonic because the tuning cohort has 257 cases; isotonic
regression is non-parametric and would overfit at that size, producing a step
function that does not transport.

Fitted on internal_val ONLY and applied unchanged elsewhere, which is the same
discipline already used for the operating threshold. No external data is used to
fit anything. Each mask variant is calibrated separately, because the two
produce different probability distributions.

What this does and does not change
----------------------------------
Platt scaling is strictly monotonic, so it cannot change the ranking of cases.
AUC, AUPRC and the ordering are therefore identical before and after. Sensitivity
and specificity at the corresponding operating point are also unchanged: the same
cases fall on the same side of the boundary. Only the probability SCALE changes,
which is what calibration metrics and decision curves depend on.

The operating threshold is re-derived on the calibrated scale, because a
threshold of 0.5054 on the raw scale corresponds to a different number after the
transform. It selects the same cases.

Output
------
    <out>/<arch>/{split}[_model]_seed{seed}_probs.txt   calibrated, same schema
    <out>/calibration.json                              parameters and checks
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402

LOCK = REPO / "protocol" / "locked_pipeline.json"
SPLITS = ["train", "internal_val", "external_test1", "external_test2"]
FIT_ON = "internal_val"


def read_probs(path: Path):
    ids, y, p = [], [], []
    with open(path) as fh:
        assert fh.readline().strip() == "case_id,label,prob_malignant,prob_benign"
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                break
            parts = line.split(",")
            if len(parts) != 4:
                break
            ids.append(parts[0]); y.append(int(parts[1])); p.append(float(parts[2]))
    return ids, np.asarray(y, int), np.asarray(p, float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path,
                    default=REPO / "protocol" / "results" / "predictions_locked")
    ap.add_argument("--out", type=Path,
                    default=REPO / "protocol" / "results" / "predictions_calibrated")
    args = ap.parse_args()

    lock = json.loads(LOCK.read_text())
    arch, seed = lock["selected_architecture"], lock["selected_seed"]
    src_dir, out_dir = args.src / arch, args.out / arch
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"locked model : {arch} seed {seed}")
    print(f"calibration  : Platt, fitted on {FIT_ON} only\n")

    record = {"architecture": arch, "seed": seed, "method": "Platt scaling",
              "fitted_on": FIT_ON, "raw_threshold": lock["threshold"],
              "variants": {}}

    for variant in ("gt", "model"):
        suffix = "" if variant == "gt" else "_model"
        fit_path = src_dir / f"{FIT_ON}{suffix}_seed{seed}_probs.txt"
        if not fit_path.exists():
            print(f"  SKIP {variant}: no {fit_path.name}")
            continue

        # ---- fit on the tuning cohort only -------------------------------
        _, y_fit, p_fit = read_probs(fit_path)
        a, b = C.calibration_slope_and_intercept_joint(y_fit, p_fit)
        print(f"[{variant}] Platt fit on {FIT_ON}: a={a:+.4f}  b={b:.4f}")

        def apply(p: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-(a + b * C.logit(p))))

        # ---- carry the locked threshold through the transform -------------
        # The operating threshold is MAPPED, not re-optimised. Re-running Youden
        # on the calibrated scale is not equivalent: threshold_youden breaks ties
        # toward prefer=0.5, and 0.5 denotes a different case split after the
        # transform. Doing so moved 43 training decisions on the model variant.
        # Calibration is meant to fix the probability scale, not to reselect an
        # operating point that was already locked in Phase D, so the locked
        # threshold is pushed through the same monotonic map as the
        # probabilities. Decisions are then identical by construction.
        thr_cal = float(1.0 / (1.0 + np.exp(-(a + b * C.logit(
            np.array([lock["threshold"]]))[0]))))

        per_split = {}
        for split in SPLITS:
            sp = src_dir / f"{split}{suffix}_seed{seed}_probs.txt"
            if not sp.exists():
                continue
            ids, y, p = read_probs(sp)
            pc = apply(p)

            with open(out_dir / sp.name, "w") as fh:
                fh.write("case_id,label,prob_malignant,prob_benign\n")
                for cid, yi, pi in zip(ids, y, pc):
                    fh.write(f"{cid},{int(yi)},{pi:.6f},{1.0 - pi:.6f}\n")

            # Monotonicity checks. Platt cannot reorder cases, so AUC must be
            # identical and the two thresholds must select the same cases. If
            # either fails, the transform was applied incorrectly.
            auc_before = C.rank_metrics(y, p)["auc"]
            auc_after = C.rank_metrics(y, pc)["auc"]
            same_decisions = bool(np.array_equal(
                (p >= lock["threshold"]).astype(int), (pc >= thr_cal).astype(int)))
            per_split[split] = {
                "n": len(y),
                "auc_before": round(auc_before, 6),
                "auc_after": round(auc_after, 6),
                "auc_unchanged": bool(abs(auc_before - auc_after) < 1e-9),
                "slope_before": round(C.calibration_slope(y, p), 4),
                "slope_after": round(C.calibration_slope(y, pc), 4),
                "brier_before": round(C.brier(y, p), 6),
                "brier_after": round(C.brier(y, pc), 6),
                "decisions_identical": same_decisions,
            }
            s = per_split[split]
            print(f"   {split:15s} slope {s['slope_before']:>6.2f} -> {s['slope_after']:>5.2f}   "
                  f"brier {s['brier_before']:.4f} -> {s['brier_after']:.4f}   "
                  f"AUC same={s['auc_unchanged']}  decisions same={s['decisions_identical']}")

        record["variants"][variant] = {
            "platt_intercept": a, "platt_slope": b,
            "calibrated_threshold": thr_cal,
            "splits": per_split,
        }
        print()

    (args.out / "calibration.json").write_text(json.dumps(record, indent=2))
    print(f"wrote {(args.out / 'calibration.json').relative_to(REPO)}")


if __name__ == "__main__":
    main()
