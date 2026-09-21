#!/usr/bin/env python3
"""
Task 6: manual (GT) mask vs automatic nnU-Net mask for densenet121.

Compares the fully automatic pipeline (nnU-Net segmentation -> DenseNet121
classification) against the same classifier fed the manual reference mask. Both
variants cover identical case sets, so the AUC comparison is paired and uses the
DeLong test for correlated ROC curves.

Each variant is evaluated at its own threshold, selected on internal_val by the
max-accuracy rule (gt -> 0.5483, model -> 0.5000). Differences in accuracy,
sensitivity and specificity get a paired bootstrap CI, resampling the same cases
for both arms.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

import common as C

ARCH = C.ARCH   # SPL_ARCH env var; defaults to densenet121 (superseded run)


def main():
    C.ensure_dirs()

    # thresholds selected on internal_val, per variant, by the pipeline's rule
    thr = {}
    for v in C.VARIANTS:
        yv, pv = C.get_y_p(ARCH, "internal_val", v)
        thr[v] = C.threshold_max_accuracy(yv, pv, prefer=0.5)

    rows = []
    for split in C.SPLITS:
        dg = C.load_probs(ARCH, split, "gt")
        dm = C.load_probs(ARCH, split, "model")
        paired = dg.case_id.tolist() == dm.case_id.tolist()
        y = dg.label.to_numpy(int)
        assert np.array_equal(y, dm.label.to_numpy(int)), f"labels differ in {split}"
        pg = dg.prob_malignant.to_numpy(float)
        pm = dm.prob_malignant.to_numpy(float)

        mg = C.threshold_metrics(y, pg, thr["gt"])
        mm = C.threshold_metrics(y, pm, thr["model"])
        aucg, aucm = float(roc_auc_score(y, pg)), float(roc_auc_score(y, pm))
        dl = C.delong_test(y, pm, pg)  # model minus gt

        # paired bootstrap for the differences (same resampled cases for both arms)
        diffs = {k: [] for k in ["auc", "accuracy", "sensitivity", "specificity"]}
        for idx in C.stratified_boot_indices(y):
            yy = y[idx]
            if len(np.unique(yy)) < 2:
                continue
            diffs["auc"].append(roc_auc_score(yy, pm[idx]) - roc_auc_score(yy, pg[idx]))
            a = C.threshold_metrics(yy, pm[idx], thr["model"])
            b = C.threshold_metrics(yy, pg[idx], thr["gt"])
            for k in ["accuracy", "sensitivity", "specificity"]:
                diffs[k].append(a[k] - b[k])

        rec = {
            "arch": ARCH, "split": split,
            "case_ids_identical_across_variants": paired,
            "n_total": len(y), "n_malignant": int((y == 1).sum()), "n_benign": int((y == 0).sum()),
            "threshold_gt": thr["gt"], "threshold_model": thr["model"],
            "auc_gt": aucg, "auc_model": aucm, "auc_diff_model_minus_gt": aucm - aucg,
            "auc_diff_delong_p": dl["p_value"],
            "auc_diff_delong_95ci_lo": dl["diff_lo"], "auc_diff_delong_95ci_hi": dl["diff_hi"],
            "auprc_gt": float(average_precision_score(y, pg)),
            "auprc_model": float(average_precision_score(y, pm)),
        }
        for k in ["accuracy", "sensitivity", "specificity", "ppv", "npv"]:
            rec[f"{k}_gt"] = mg[k]
            rec[f"{k}_model"] = mm[k]
            rec[f"{k}_diff_model_minus_gt"] = mm[k] - mg[k]
            dn_g, dn_m = mg[C.METRIC_DENOM_KEY[k]], mm[C.METRIC_DENOM_KEY[k]]
            rec[f"{k}_denom_n_gt"] = dn_g
            rec[f"{k}_denom_n_model"] = dn_m
            lo, hi = C.wilson_ci(mg[k], dn_g)
            rec[f"{k}_gt_wilson_lo"], rec[f"{k}_gt_wilson_hi"] = lo, hi
            lo, hi = C.wilson_ci(mm[k], dn_m)
            rec[f"{k}_model_wilson_lo"], rec[f"{k}_model_wilson_hi"] = lo, hi
        for k, vals in diffs.items():
            v = np.asarray(vals)
            rec[f"{k}_diff_boot_lo"] = float(np.percentile(v, 2.5))
            rec[f"{k}_diff_boot_hi"] = float(np.percentile(v, 97.5))
        rec["tp_gt"], rec["fp_gt"], rec["tn_gt"], rec["fn_gt"] = mg["tp"], mg["fp"], mg["tn"], mg["fn"]
        rec["tp_model"], rec["fp_model"], rec["tn_model"], rec["fn_model"] = mm["tp"], mm["fp"], mm["tn"], mm["fn"]
        rows.append(rec)

    df = pd.DataFrame(rows)
    df["split"] = pd.Categorical(df["split"], C.SPLITS, ordered=True)
    df = df.sort_values("split").reset_index(drop=True)
    df.to_csv(C.RESULTS / "task6_gt_vs_automatic_mask.csv", index=False)

    print("=== Task 6: manual (GT) vs automatic nnU-Net mask, DenseNet121 ===")
    print(f"Case ids identical across variants in every split: {bool(df.case_ids_identical_across_variants.all())}")
    show = df.copy()
    for k in ["accuracy", "sensitivity", "specificity"]:
        show[f"{k} gt"] = show.apply(lambda r: f"{r[k+'_gt']:.3f} ({r[k+'_gt_wilson_lo']:.3f}-{r[k+'_gt_wilson_hi']:.3f})", axis=1)
        show[f"{k} auto"] = show.apply(lambda r: f"{r[k+'_model']:.3f} ({r[k+'_model_wilson_lo']:.3f}-{r[k+'_model_wilson_hi']:.3f})", axis=1)
    show["AUC gt"] = show.auc_gt.round(3)
    show["AUC auto"] = show.auc_model.round(3)
    show["dAUC (auto-gt)"] = show.apply(
        lambda r: f"{r.auc_diff_model_minus_gt:+.3f} (DeLong p={r.auc_diff_delong_p:.3f})", axis=1)
    cols = ["split", "n_total", "n_malignant", "n_benign", "AUC gt", "AUC auto", "dAUC (auto-gt)",
            "accuracy gt", "accuracy auto", "sensitivity gt", "sensitivity auto",
            "specificity gt", "specificity auto"]
    print(show[cols].to_string(index=False))


if __name__ == "__main__":
    main()
