#!/usr/bin/env python3
"""
Task 1: confidence intervals for the densenet121 metrics.

Part A takes two worked examples (internal-validation sensitivity and External
Test 2 specificity) and computes the Wilson interval two ways, on the whole
cohort n and on the class-specific denominator, then locates the operating point
in the locked dataset that produces the reported point estimates.

Part B rebuilds the full densenet121 metric table for both mask variants and
every cohort using the class-specific denominators, with Wilson and stratified
case-level bootstrap percentile CIs, and DeLong plus bootstrap CIs for AUC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

import common as C

ARCH = "densenet121"

# Operating thresholds used by the pipeline: selected on internal_val by maximum
# accuracy, then applied unchanged to every other split. Taken from
# binary_classification/predictions_tight/grand_summary.txt.
LOCKED_THR = {"gt": 0.5483, "model": 0.5000}


# ---------------------------------------------------------------------------
# Part A - denominator check on two worked examples
# ---------------------------------------------------------------------------

def part_a():
    rows = []

    # ---- example 1: internal validation sensitivity ----
    # 0.90 (0.86-0.93) is Wilson on 232/257; on the class-specific denominator it
    # is Wilson on 140/155. Find the mask variant and threshold yielding 140/155.
    hits = []
    for variant in C.VARIANTS:
        y, p = C.get_y_p(ARCH, "internal_val", variant)
        thr = LOCKED_THR[variant]
        m = C.threshold_metrics(y, p, thr)
        hits.append((variant, thr, m))

    for variant, thr, m in hits:
        sens = m["sensitivity"]
        n_pos, n_all = m["n_pos"], m["n"]
        wrong_lo, wrong_hi = C.wilson_ci(sens, n_all)
        right_lo, right_hi = C.wilson_ci(sens, n_pos)
        rows.append({
            "quantity": "internal_val sensitivity",
            "mask_variant": variant, "threshold": thr,
            "point_estimate": sens,
            "numerator": m["tp"], "correct_denominator": n_pos,
            "whole_cohort_denominator": n_all,
            "wilson_on_whole_cohort_lo": wrong_lo, "wilson_on_whole_cohort_hi": wrong_hi,
            "wilson_on_correct_denom_lo": right_lo, "wilson_on_correct_denom_hi": right_hi,
            "implied_whole_cohort_numerator_round": int(round(sens * n_all)),
            "ci_on_cohort_n": "0.86-0.93",
            "ci_on_class_n": "0.846-0.940",
            "expected_fraction": "140/155",
            "fraction_matches": (m["tp"] == 140 and n_pos == 155),
        })

    # ---- example 2: External Test 2 specificity ----
    # 0.67 (0.57-0.75) is Wilson on 63/94; on the class-specific denominator it
    # is Wilson on 18/27.
    for variant in C.VARIANTS:
        y, p = C.get_y_p(ARCH, "external_test2", variant)
        thr = LOCKED_THR[variant]
        m = C.threshold_metrics(y, p, thr)
        spec = m["specificity"]
        n_neg, n_all = m["n_neg"], m["n"]
        wrong_lo, wrong_hi = C.wilson_ci(spec, n_all)
        right_lo, right_hi = C.wilson_ci(spec, n_neg)
        rows.append({
            "quantity": "external_test2 specificity",
            "mask_variant": variant, "threshold": thr,
            "point_estimate": spec,
            "numerator": m["tn"], "correct_denominator": n_neg,
            "whole_cohort_denominator": n_all,
            "wilson_on_whole_cohort_lo": wrong_lo, "wilson_on_whole_cohort_hi": wrong_hi,
            "wilson_on_correct_denom_lo": right_lo, "wilson_on_correct_denom_hi": right_hi,
            "implied_whole_cohort_numerator_round": int(round(spec * n_all)),
            "ci_on_cohort_n": "0.57-0.75",
            "ci_on_class_n": "0.478-0.814",
            "expected_fraction": "18/27",
            "fraction_matches": (m["tn"] == 18 and n_neg == 27),
        })

    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "task1a_ci_denominator_check.csv", index=False)

    # The same Wilson arithmetic on bare counts, independent of the loaded data.
    lit = []
    for name, x, n, claim in [
        ("Wilson 232/257 (alleged wrong internal_val sensitivity CI)", 232, 257, "0.86-0.93"),
        ("Wilson 140/155 (correct internal_val sensitivity CI)", 140, 155, "0.846-0.940"),
        ("Wilson 63/94 (alleged wrong ExtTest2 specificity CI)", 63, 94, "0.57-0.75"),
        ("Wilson 18/27 (correct ExtTest2 specificity CI)", 18, 27, "0.478-0.814"),
    ]:
        lo, hi = C.wilson_from_counts(x, n)
        lit.append({"quantity": name, "x": x, "n": n, "p_hat": x / n,
                    "wilson_lo": lo, "wilson_hi": hi,
                    "published_value": claim,
                    "reproduced": f"{lo:.3f}-{hi:.3f}"})
    lit_df = pd.DataFrame(lit)
    lit_df.to_csv(C.RESULTS / "task1a_wilson_arithmetic_check.csv", index=False)

    print("=== Task 1A: Wilson intervals by denominator ===")
    print(lit_df.to_string(index=False))
    print("\n=== Task 1A: which locked operating point produces those point estimates ===")
    print(df[["quantity", "mask_variant", "threshold", "point_estimate", "numerator",
              "correct_denominator", "whole_cohort_denominator",
              "fraction_matches"]].to_string(index=False))
    print("\nCIs at the matching operating point:")
    print(df[df.fraction_matches][
        ["quantity", "mask_variant", "point_estimate",
         "wilson_on_whole_cohort_lo", "wilson_on_whole_cohort_hi",
         "wilson_on_correct_denom_lo", "wilson_on_correct_denom_hi"]].round(4).to_string(index=False))
    return df, lit_df


# ---------------------------------------------------------------------------
# Part B - full metric table
# ---------------------------------------------------------------------------

def part_b():
    recs = []
    for variant in C.VARIANTS:
        thr = LOCKED_THR[variant]
        for split in C.SPLITS:
            y, p = C.get_y_p(ARCH, split, variant)
            m = C.threshold_metrics(y, p, thr)
            n_pos, n_neg, n_all = m["n_pos"], m["n_neg"], m["n"]

            base = {
                "arch": ARCH, "mask_variant": variant, "split": split,
                "threshold": thr, "n_total": n_all,
                "n_malignant": n_pos, "n_benign": n_neg,
                "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
            }

            # --- AUC ---
            dl = C.delong_auc_ci(y, p)
            bl, bh, _ = C.bootstrap_ci(y, p, lambda yy, pp: roc_auc_score(yy, pp))
            recs.append({**base, "metric": "AUC",
                         "numerator": np.nan, "denominator": f"{n_pos} malignant vs {n_neg} benign",
                         "denominator_n": np.nan,
                         "estimate": dl["auc"],
                         "wilson_lo": np.nan, "wilson_hi": np.nan,
                         "delong_lo": dl["delong_lo"], "delong_hi": dl["delong_hi"],
                         "delong_logit_lo": dl["delong_logit_lo"], "delong_logit_hi": dl["delong_logit_hi"],
                         "delong_se": dl["se"],
                         "boot_lo": bl, "boot_hi": bh,
                         "whole_cohort_wilson_lo": np.nan, "whole_cohort_wilson_hi": np.nan})

            # --- AUPRC / AP ---
            ap = float(average_precision_score(y, p))
            bl, bh, _ = C.bootstrap_ci(y, p, lambda yy, pp: average_precision_score(yy, pp))
            recs.append({**base, "metric": "AUPRC",
                         "numerator": np.nan, "denominator": f"{n_pos} malignant vs {n_neg} benign",
                         "denominator_n": np.nan,
                         "estimate": ap,
                         "wilson_lo": np.nan, "wilson_hi": np.nan,
                         "delong_lo": np.nan, "delong_hi": np.nan,
                         "delong_logit_lo": np.nan, "delong_logit_hi": np.nan, "delong_se": np.nan,
                         "boot_lo": bl, "boot_hi": bh,
                         "whole_cohort_wilson_lo": np.nan, "whole_cohort_wilson_hi": np.nan})

            # --- proportion metrics, on their class-specific denominators ---
            num_key = {"accuracy": None, "sensitivity": "tp", "specificity": "tn",
                       "ppv": "tp", "npv": "tn"}
            for name in ["accuracy", "sensitivity", "specificity", "ppv", "npv"]:
                est = m[name]
                dkey = C.METRIC_DENOM_KEY[name]
                dn = m[dkey]
                num = (m["tp"] + m["tn"]) if name == "accuracy" else m[num_key[name]]
                wlo, whi = C.wilson_ci(est, dn)
                # same estimate paired with the whole-cohort n, for comparison
                cwlo, cwhi = C.wilson_ci(est, n_all)

                def stat(yy, pp, _name=name):
                    mm = C.threshold_metrics(yy, pp, thr)
                    return mm[_name]

                bl, bh, _ = C.bootstrap_ci(y, p, stat)
                recs.append({**base, "metric": name,
                             "numerator": num,
                             "denominator": {"accuracy": "all cases",
                                             "sensitivity": "malignant cases",
                                             "specificity": "benign cases",
                                             "ppv": "test-positive cases",
                                             "npv": "test-negative cases"}[name],
                             "denominator_n": dn,
                             "estimate": est,
                             "wilson_lo": wlo, "wilson_hi": whi,
                             "delong_lo": np.nan, "delong_hi": np.nan,
                             "delong_logit_lo": np.nan, "delong_logit_hi": np.nan, "delong_se": np.nan,
                             "boot_lo": bl, "boot_hi": bh,
                             "whole_cohort_wilson_lo": cwlo, "whole_cohort_wilson_hi": cwhi})

    df = pd.DataFrame(recs)
    order_metric = ["AUC", "AUPRC", "accuracy", "sensitivity", "specificity", "ppv", "npv"]
    df["metric"] = pd.Categorical(df["metric"], order_metric, ordered=True)
    df["split"] = pd.Categorical(df["split"], C.SPLITS, ordered=True)
    df = df.sort_values(["mask_variant", "split", "metric"]).reset_index(drop=True)
    df.to_csv(C.RESULTS / "task1b_densenet121_corrected_metrics_with_ci.csv", index=False)

    print("\n=== Task 1B: corrected densenet121 metrics (locked operating threshold) ===")
    show = df[df.split != "train"].copy()
    show["est(95% Wilson)"] = show.apply(
        lambda r: f"{r.estimate:.3f} ({r.wilson_lo:.3f}-{r.wilson_hi:.3f})"
        if np.isfinite(r.wilson_lo) else f"{r.estimate:.3f} (-)", axis=1)
    show["est(95% boot)"] = show.apply(
        lambda r: f"{r.estimate:.3f} ({r.boot_lo:.3f}-{r.boot_hi:.3f})", axis=1)
    show["denom"] = show.apply(
        lambda r: f"{int(r.denominator_n)} {r.denominator}" if np.isfinite(r.denominator_n)
        else r.denominator, axis=1)
    print(show[["mask_variant", "split", "metric", "denom",
                "est(95% Wilson)", "est(95% boot)"]].to_string(index=False))
    return df


def main():
    C.ensure_dirs()
    part_a()
    part_b()


if __name__ == "__main__":
    main()
