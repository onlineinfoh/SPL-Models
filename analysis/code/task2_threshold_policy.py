#!/usr/bin/env python3
"""
Task 2: threshold policy comparison for densenet121.

Three policies are compared:
  (1) fixed 0.5, as hard-coded in binary_classification/train.py for its
      per-epoch metrics
  (2) maximum accuracy on internal_val, as used by
      binary_classification/infer_probs_tight.py::_best_threshold_from_rows
  (3) maximum Youden's J on internal_val

Every threshold is derived on internal_val and then applied unchanged to the
other splits. The last block checks how often the two data-driven policies pick
the same threshold across all 13 architectures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as C

ARCH = "densenet121"
POLICIES = ["fixed_0.5", "max_accuracy", "youden_J"]


def derive_thresholds(variant: str) -> dict:
    y, p = C.get_y_p(ARCH, "internal_val", variant)
    return {
        "fixed_0.5": 0.5,
        "max_accuracy": C.threshold_max_accuracy(y, p, prefer=0.5),
        "youden_J": C.threshold_youden(y, p, prefer=0.5),
    }


def main():
    C.ensure_dirs()
    rows = []
    thr_rows = []
    for variant in C.VARIANTS:
        thrs = derive_thresholds(variant)
        yv, pv = C.get_y_p(ARCH, "internal_val", variant)
        for policy, thr in thrs.items():
            mv = C.threshold_metrics(yv, pv, thr)
            thr_rows.append({
                "arch": ARCH, "mask_variant": variant, "policy": policy,
                "threshold": thr,
                "tuning_cohort": "internal_val",
                "tuning_accuracy": mv["accuracy"],
                "tuning_sensitivity": mv["sensitivity"],
                "tuning_specificity": mv["specificity"],
                "tuning_youden_J": mv["youden_j"],
            })
            for split in C.SPLITS:
                y, p = C.get_y_p(ARCH, split, variant)
                m = C.threshold_metrics(y, p, thr)
                rec = {
                    "arch": ARCH, "mask_variant": variant, "policy": policy,
                    "threshold": thr, "split": split,
                    "n_total": m["n"], "n_malignant": m["n_pos"], "n_benign": m["n_neg"],
                    "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
                }
                for name in ["accuracy", "sensitivity", "specificity", "ppv", "npv", "youden_j", "f1"]:
                    rec[name] = m[name]
                    if name in C.METRIC_DENOM_KEY:
                        dn = m[C.METRIC_DENOM_KEY[name]]
                        lo, hi = C.wilson_ci(m[name], dn)
                        rec[f"{name}_denom_n"] = dn
                        rec[f"{name}_wilson_lo"] = lo
                        rec[f"{name}_wilson_hi"] = hi
                rows.append(rec)

    thr_df = pd.DataFrame(thr_rows)
    thr_df.to_csv(C.RESULTS / "task2_thresholds_derived_on_internal_val.csv", index=False)

    df = pd.DataFrame(rows)
    df["split"] = pd.Categorical(df["split"], C.SPLITS, ordered=True)
    df["policy"] = pd.Categorical(df["policy"], POLICIES, ordered=True)
    df = df.sort_values(["mask_variant", "split", "policy"]).reset_index(drop=True)
    df.to_csv(C.RESULTS / "task2_threshold_policy_comparison.csv", index=False)

    # deltas relative to the max_accuracy policy
    piv = df.pivot_table(index=["mask_variant", "split"], columns="policy",
                         values=["accuracy", "sensitivity", "specificity"], observed=False)
    deltas = []
    for (variant, split), _ in piv.iterrows():
        row = {"mask_variant": variant, "split": split}
        for met in ["accuracy", "sensitivity", "specificity"]:
            row[f"{met}_used_max_acc"] = piv.loc[(variant, split), (met, "max_accuracy")]
            row[f"{met}_youden"] = piv.loc[(variant, split), (met, "youden_J")]
            row[f"{met}_delta_youden_minus_used"] = row[f"{met}_youden"] - row[f"{met}_used_max_acc"]
            row[f"{met}_fixed05"] = piv.loc[(variant, split), (met, "fixed_0.5")]
        deltas.append(row)
    dd = pd.DataFrame(deltas)
    dd.to_csv(C.RESULTS / "task2_youden_vs_used_deltas.csv", index=False)

    print("=== Task 2: thresholds derived on internal_val only ===")
    print(thr_df.round(4).to_string(index=False))
    print("\n=== Task 2: metrics under each policy ===")
    print(df[["mask_variant", "split", "policy", "threshold", "accuracy",
              "sensitivity", "specificity", "ppv", "npv"]].round(4).to_string(index=False))
    print("\n=== Task 2: how much would switching to Youden move the numbers? ===")
    print(dd.round(4).to_string(index=False))

    # Max-accuracy and Youden are different criteria that happen to coincide for
    # densenet121. Repeat the comparison over all architectures.
    arch_rows = []
    for variant in C.VARIANTS:
        for a in C.ARCHS:
            y, p = C.get_y_p(a, "internal_val", variant)
            ta = C.threshold_max_accuracy(y, p, prefer=0.5)
            ty = C.threshold_youden(y, p, prefer=0.5)
            ma, my = C.threshold_metrics(y, p, ta), C.threshold_metrics(y, p, ty)
            arch_rows.append({
                "mask_variant": variant, "arch": a,
                "thr_max_accuracy": ta, "thr_youden": ty,
                "thresholds_identical": bool(abs(ta - ty) < 1e-9),
                "internal_val_acc_at_max_acc_thr": ma["accuracy"],
                "internal_val_acc_at_youden_thr": my["accuracy"],
                "internal_val_sens_at_max_acc_thr": ma["sensitivity"],
                "internal_val_sens_at_youden_thr": my["sensitivity"],
                "internal_val_spec_at_max_acc_thr": ma["specificity"],
                "internal_val_spec_at_youden_thr": my["specificity"],
            })
    ar = pd.DataFrame(arch_rows)
    ar.to_csv(C.RESULTS / "task2_maxacc_vs_youden_all_architectures.csv", index=False)
    n_same = int(ar[ar.mask_variant == "gt"].thresholds_identical.sum())
    print(f"\n=== Task 2: max-accuracy vs Youden thresholds across all 13 architectures (gt masks) ===")
    print(ar[ar.mask_variant == "gt"][
        ["arch", "thr_max_accuracy", "thr_youden", "thresholds_identical"]].round(4).to_string(index=False))
    print(f"Identical for {n_same}/13 architectures - so the two criteria are genuinely different, "
          f"they merely coincide for densenet121.")


if __name__ == "__main__":
    main()
