#!/usr/bin/env python3
# ==========================================================================
# SUPERSEDED BY THE LOCKED RE-ANALYSIS
# ==========================================================================
#
# CANNOT be run under the locked protocol: it scores all 13 architectures on
# external data, which the hold-out permits only for the selected model.
# Replaced by the internal-only ranking in protocol/locked_pipeline.json.
#
# Retained unmodified as the audit record. Produces no reported result.
# See README.md and docs/REPRODUCE.md for the active pipeline.
# ==========================================================================
#
"""
Task 5: architecture ranking and pairwise AUC comparisons.

The selection criterion in infer_probs_tight.py is: for each architecture pick
the threshold maximising internal_val accuracy, then rank architectures by that
accuracy. This reproduces that ranking and adds the threshold-free ranking by
internal_val AUC, plus DeLong tests of densenet121 against efficientnet_b1 on
every split and against every other architecture on external_test1.

The internal_val accuracy column is a within-sample quantity, since the
threshold is tuned on the same cohort. The AUC column is threshold-free but is
still a selection statistic over 13 candidates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

import common as C


def main():
    C.ensure_dirs()

    rows = []
    for variant in C.VARIANTS:
        for arch in C.ARCHS:
            yv, pv = C.get_y_p(arch, "internal_val", variant)
            thr = C.threshold_max_accuracy(yv, pv, prefer=0.5)
            mv = C.threshold_metrics(yv, pv, thr)
            auc_v = float(roc_auc_score(yv, pv))
            dl_v = C.delong_auc_ci(yv, pv)

            rec = {
                "mask_variant": variant, "arch": arch,
                "internal_val_threshold_max_acc": thr,
                "internal_val_accuracy": mv["accuracy"],
                "internal_val_sensitivity": mv["sensitivity"],
                "internal_val_specificity": mv["specificity"],
                "internal_val_auc": auc_v,
                "internal_val_auc_delong_lo": dl_v["delong_lo"],
                "internal_val_auc_delong_hi": dl_v["delong_hi"],
                "internal_val_auprc": float(average_precision_score(yv, pv)),
            }
            # externally applied, threshold held fixed
            for split in ["external_test1", "external_test2"]:
                y, p = C.get_y_p(arch, split, variant)
                m = C.threshold_metrics(y, p, thr)
                rec[f"{split}_auc"] = float(roc_auc_score(y, p))
                rec[f"{split}_accuracy"] = m["accuracy"]
                rec[f"{split}_sensitivity"] = m["sensitivity"]
                rec[f"{split}_specificity"] = m["specificity"]
            rows.append(rec)

    df = pd.DataFrame(rows)
    for variant in C.VARIANTS:
        sel = df.mask_variant == variant
        df.loc[sel, "rank_by_internal_val_accuracy"] = (
            df.loc[sel, "internal_val_accuracy"].rank(ascending=False, method="min").astype(int))
        df.loc[sel, "rank_by_internal_val_auc"] = (
            df.loc[sel, "internal_val_auc"].rank(ascending=False, method="min").astype(int))
        df.loc[sel, "rank_by_external_test1_auc"] = (
            df.loc[sel, "external_test1_auc"].rank(ascending=False, method="min").astype(int))
        df.loc[sel, "rank_by_external_test2_auc"] = (
            df.loc[sel, "external_test2_auc"].rank(ascending=False, method="min").astype(int))
    df = df.sort_values(["mask_variant", "rank_by_internal_val_accuracy"]).reset_index(drop=True)
    df.to_csv(C.RESULTS / "task5_architecture_ranking.csv", index=False)

    for variant in C.VARIANTS:
        print(f"\n=== Task 5: architecture ranking, mask variant = {variant} ===")
        sub = df[df.mask_variant == variant]
        print(sub[["rank_by_internal_val_accuracy", "arch", "internal_val_threshold_max_acc",
                   "internal_val_accuracy", "internal_val_auc", "rank_by_internal_val_auc",
                   "external_test1_auc", "rank_by_external_test1_auc",
                   "external_test2_auc"]].round(4).to_string(index=False))

    # ---- DeLong: densenet121 vs efficientnet_b1 ----
    tests = []
    for variant in C.VARIANTS:
        for split in C.SPLITS:
            y1, p1 = C.get_y_p("densenet121", split, variant)
            y2, p2 = C.get_y_p("efficientnet_b1", split, variant)
            assert np.array_equal(y1, y2), f"label vectors differ for {split}/{variant}"
            r = C.delong_test(y1, p1, p2)
            tests.append({
                "mask_variant": variant, "split": split,
                "n_total": len(y1), "n_malignant": int((y1 == 1).sum()),
                "n_benign": int((y1 == 0).sum()),
                "auc_densenet121": r["auc1"], "auc_efficientnet_b1": r["auc2"],
                "auc_diff_densenet_minus_effb1": r["auc_diff"],
                "se_diff": r["se_diff"], "z": r["z"],
                "delong_p_value": r["p_value"],
                "diff_95ci_lo": r["diff_lo"], "diff_95ci_hi": r["diff_hi"],
            })
    tdf = pd.DataFrame(tests)
    tdf["split"] = pd.Categorical(tdf["split"], C.SPLITS, ordered=True)
    tdf = tdf.sort_values(["mask_variant", "split"]).reset_index(drop=True)
    tdf.to_csv(C.RESULTS / "task5_delong_densenet121_vs_efficientnetb1.csv", index=False)

    print("\n=== Task 5: DeLong, densenet121 vs efficientnet_b1 (paired, same cases) ===")
    print(tdf.round(4).to_string(index=False))

    # ---- DeLong vs densenet121 for every architecture on external_test1 ----
    ctx = []
    for variant in C.VARIANTS:
        y, p_d = C.get_y_p("densenet121", "external_test1", variant)
        for arch in C.ARCHS:
            if arch == "densenet121":
                continue
            y2, p2 = C.get_y_p(arch, "external_test1", variant)
            r = C.delong_test(y, p_d, p2)
            ctx.append({"mask_variant": variant, "comparator": arch,
                        "auc_densenet121": r["auc1"], "auc_comparator": r["auc2"],
                        "auc_diff": r["auc_diff"], "delong_p_value": r["p_value"]})
    cdf = pd.DataFrame(ctx)
    cdf.to_csv(C.RESULTS / "task5_delong_external_test1_all_vs_densenet121.csv", index=False)
    print("\n=== Task 5: external_test1 DeLong of densenet121 vs every other architecture (gt masks) ===")
    print(cdf[cdf.mask_variant == "gt"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
