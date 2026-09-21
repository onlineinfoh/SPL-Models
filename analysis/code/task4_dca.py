#!/usr/bin/env python3
"""
Task 4: decision-curve analysis for densenet121.

    NB_model(pt) = TP/n - (FP/n) * (pt / (1 - pt)),  classifying p >= pt positive
    NB_all(pt)   = prev - (1 - prev) * (pt / (1 - pt))
    NB_none(pt)  = 0

The treat-all curve decreases monotonically in pt and crosses zero at pt = prev;
the prevalence is only its limiting value as pt -> 0, so it is not a horizontal
line. Both forms are written out for comparison.

The single-threshold net benefit at pt = 0.50 is reported separately from the
mean over the pt grid, since the two are different quantities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as C

ARCH = C.ARCH   # SPL_ARCH env var; defaults to densenet121 (superseded run)
FIG_SPLITS = ["internal_val", "external_test1", "external_test2"]
PT_GRID = np.round(np.arange(0.01, 0.8001, 0.01), 4)


def main():
    C.ensure_dirs()
    plt = C.set_style()

    curve_rows = []
    for variant in C.VARIANTS:
        for split in C.SPLITS:
            y, p = C.get_y_p(ARCH, split, variant)
            prev = float(np.mean(y))
            n = len(y)
            for pt in PT_GRID:
                nb_m = C.net_benefit_model(y, p, float(pt))
                nb_all = C.net_benefit_treat_all(y, float(pt))
                tp, fp, tn, fn = C.confusion(y, p, float(pt))
                curve_rows.append({
                    "arch": ARCH, "mask_variant": variant, "split": split,
                    "n_total": n, "n_malignant": int((y == 1).sum()),
                    "n_benign": int((y == 0).sum()), "prevalence": prev,
                    "threshold_probability_pt": float(pt),
                    "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                    "net_benefit_model": nb_m,
                    "net_benefit_treat_all_correct": nb_all,
                    "net_benefit_treat_all_WRONG_flat_at_prevalence": prev,
                    "net_benefit_treat_none": 0.0,
                    "delta_nb_model_minus_treat_all": nb_m - nb_all,
                    "standardised_net_benefit_model": nb_m / prev if prev > 0 else np.nan,
                })
    cdf = pd.DataFrame(curve_rows)
    cdf.to_csv(C.RESULTS / "task4_dca_curves.csv", index=False)

    # ---- explicit single-threshold summary at pt = 0.50 ----
    pt0 = 0.50
    rows = []
    for variant in C.VARIANTS:
        for split in C.SPLITS:
            y, p = C.get_y_p(ARCH, split, variant)
            n = len(y)
            prev = float(np.mean(y))
            tp, fp, tn, fn = C.confusion(y, p, pt0)
            nb_m = C.net_benefit_model(y, p, pt0)
            nb_all = C.net_benefit_treat_all(y, pt0)

            lo, hi, _ = C.bootstrap_ci(y, p, lambda yy, pp: C.net_benefit_model(yy, pp, pt0))
            # the whole grid, for the mean-over-pt column below
            sub = cdf[(cdf.mask_variant == variant) & (cdf.split == split)]
            rows.append({
                "arch": ARCH, "mask_variant": variant, "split": split,
                "n_total": n, "n_malignant": int((y == 1).sum()), "n_benign": int((y == 0).sum()),
                "prevalence": prev,
                "threshold_probability_pt": pt0,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "net_benefit_model_AT_pt_0.50": nb_m,
                "net_benefit_model_AT_pt_0.50_boot_lo": lo,
                "net_benefit_model_AT_pt_0.50_boot_hi": hi,
                "net_benefit_treat_all_AT_pt_0.50_correct": nb_all,
                "net_benefit_treat_all_AT_pt_0.50_WRONG_flat": prev,
                "incremental_nb_vs_treat_all_AT_pt_0.50": nb_m - nb_all,
                "net_true_positives_per_100_vs_treat_all_AT_pt_0.50":
                    100 * (nb_m - nb_all) / (pt0 / (1 - pt0)),
                "mean_net_benefit_over_pt_0.01_0.80_NOT_the_same_thing":
                    float(sub.net_benefit_model.mean()),
                "pt_range_where_model_beats_both_references":
                    _beats_range(sub),
            })
    sdf = pd.DataFrame(rows)
    sdf["split"] = pd.Categorical(sdf["split"], C.SPLITS, ordered=True)
    sdf = sdf.sort_values(["mask_variant", "split"]).reset_index(drop=True)
    sdf.to_csv(C.RESULTS / "task4_dca_net_benefit_at_pt050.csv", index=False)

    print("=== Task 4: net benefit AT THRESHOLD PROBABILITY pt = 0.50 (not an average) ===")
    print(sdf[["mask_variant", "split", "prevalence",
               "net_benefit_model_AT_pt_0.50",
               "net_benefit_treat_all_AT_pt_0.50_correct",
               "net_benefit_treat_all_AT_pt_0.50_WRONG_flat",
               "incremental_nb_vs_treat_all_AT_pt_0.50",
               "mean_net_benefit_over_pt_0.01_0.80_NOT_the_same_thing"]].round(4).to_string(index=False))
    print("\npt range where the model curve is above BOTH reference strategies:")
    print(sdf[["mask_variant", "split", "pt_range_where_model_beats_both_references"]].to_string(index=False))

    # ---------------- figures ----------------
    for split in FIG_SPLITS:
        fig, ax = plt.subplots(figsize=(5.4, 4.3))
        sub_gt = cdf[(cdf.split == split) & (cdf.mask_variant == "gt")]
        prev = float(sub_gt.prevalence.iloc[0])

        ax.axhline(0.0, color=C.CB["grey"], lw=1.3, ls="-", zorder=1,
                   label="Treat none (net benefit = 0)")
        ax.plot(sub_gt.threshold_probability_pt, sub_gt.net_benefit_treat_all_correct,
                color=C.CB["black"], lw=1.5, ls=(0, (5, 2)), zorder=2,
                label=r"Treat all: $prev-(1-prev)\,p_t/(1-p_t)$")
        ax.axhline(prev, color=C.CB["purple"], lw=1.2, ls=(0, (1, 2)), zorder=2,
                   label=f"Treat all drawn as a flat line at prevalence\n"
                         f"({prev:.3f}) - the incorrect version")

        for variant, col, ls in [("gt", C.CB["blue"], "-"), ("model", C.CB["vermillion"], "-")]:
            s = cdf[(cdf.split == split) & (cdf.mask_variant == variant)]
            lbl = "DenseNet121, manual (reference) mask" if variant == "gt" \
                else "DenseNet121, automatic nnU-Net mask"
            ax.plot(s.threshold_probability_pt, s.net_benefit_model, color=col, ls=ls,
                    lw=2.0, zorder=4, label=lbl)

        ax.axvline(0.50, color=C.CB["grey"], lw=0.8, ls=(0, (1, 3)), zorder=1)
        ax.annotate(r"$p_t=0.50$", xy=(0.505, 0.02), xycoords=("data", "axes fraction"),
                    fontsize=7.5, color=C.CB["grey"], rotation=90, va="bottom")

        ax.set_xlim(0.0, 0.80)
        lo = min(-0.05, float(cdf[(cdf.split == split)].net_benefit_model.min()) - 0.03)
        ax.set_ylim(max(lo, -0.25), max(prev, float(sub_gt.net_benefit_model.max())) * 1.12)
        ax.set_xlabel("Threshold probability $p_t$")
        ax.set_ylabel("Net benefit")
        n_all = int(sub_gt.n_total.iloc[0])
        npos = int(sub_gt.n_malignant.iloc[0])
        nneg = int(sub_gt.n_benign.iloc[0])
        ax.set_title(f"Decision curve analysis - {C.PRETTY_SPLIT[split]}\n"
                     f"n = {n_all} ({npos} malignant / {nneg} benign), prevalence = {prev:.3f}",
                     pad=8)
        ax.legend(loc="lower left", fontsize=7.3, labelspacing=0.55)
        C.save_fig(fig, f"task4_dca_{split}")
        plt.close(fig)


def _beats_range(sub: pd.DataFrame) -> str:
    """Contiguous-ish pt range where NB_model exceeds both treat-all and treat-none."""
    ok = sub[(sub.net_benefit_model > sub.net_benefit_treat_all_correct) &
             (sub.net_benefit_model > 0)]
    if ok.empty:
        return "none"
    return f"{ok.threshold_probability_pt.min():.2f}-{ok.threshold_probability_pt.max():.2f}"


if __name__ == "__main__":
    main()
