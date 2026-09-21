#!/usr/bin/env python3
"""
Task 3: calibration of densenet121, per cohort and mask variant.

Reports the Brier score, calibration-in-the-large (intercept of
y ~ 1 + offset(logit(p)), slope fixed at 1), the calibration slope (b from
y ~ a + b*logit(p)), and a 10-bin quantile calibration curve with per-bin counts
and Wilson intervals. The three summary statistics get stratified case-level
bootstrap percentile CIs.

Perfect calibration is intercept 0 and slope 1. Slope below 1 means predictions
are too extreme; intercept above 0 means risk is systematically under-predicted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import common as C

ARCH = C.ARCH   # SPL_ARCH env var; defaults to densenet121 (superseded run)
FIG_SPLITS = ["internal_val", "external_test1", "external_test2"]


def main():
    C.ensure_dirs()
    plt = C.set_style()

    summary = []
    curves = []
    for variant in C.VARIANTS:
        for split in C.SPLITS:
            y, p = C.get_y_p(ARCH, split, variant)
            n_pos = int((y == 1).sum())
            n_neg = int((y == 0).sum())

            b = C.brier(y, p)
            icpt = C.calibration_intercept(y, p)
            slope = C.calibration_slope(y, p)

            b_lo, b_hi, _ = C.bootstrap_ci(y, p, C.brier)
            i_lo, i_hi, _ = C.bootstrap_ci(y, p, C.calibration_intercept)
            s_lo, s_hi, _ = C.bootstrap_ci(y, p, C.calibration_slope)

            # Brier decomposition reference: the "null" Brier at the observed prevalence
            prev = float(np.mean(y))
            brier_null = float(np.mean((prev - y) ** 2))

            summary.append({
                "arch": ARCH, "mask_variant": variant, "split": split,
                "n_total": len(y), "n_malignant": n_pos, "n_benign": n_neg,
                "prevalence": prev,
                "brier": b, "brier_boot_lo": b_lo, "brier_boot_hi": b_hi,
                "brier_null_at_prevalence": brier_null,
                "brier_skill_score_vs_null": 1 - b / brier_null if brier_null > 0 else np.nan,
                "calibration_intercept": icpt,
                "calibration_intercept_boot_lo": i_lo, "calibration_intercept_boot_hi": i_hi,
                "calibration_slope": slope,
                "calibration_slope_boot_lo": s_lo, "calibration_slope_boot_hi": s_hi,
                "mean_predicted_risk": float(np.mean(p)),
                "observed_risk": prev,
            })

            cc = C.calibration_curve_quantile(y, p, n_bins=10)
            cc.insert(0, "split", split)
            cc.insert(0, "mask_variant", variant)
            cc.insert(0, "arch", ARCH)
            curves.append(cc)

    sdf = pd.DataFrame(summary)
    sdf["split"] = pd.Categorical(sdf["split"], C.SPLITS, ordered=True)
    sdf = sdf.sort_values(["mask_variant", "split"]).reset_index(drop=True)
    sdf.to_csv(C.RESULTS / "task3_calibration_summary.csv", index=False)

    cdf = pd.concat(curves, ignore_index=True)
    cdf.to_csv(C.RESULTS / "task3_calibration_curves_10bin.csv", index=False)

    print("=== Task 3: calibration summary (densenet121) ===")
    disp = sdf.copy()
    disp["Brier (95% boot)"] = disp.apply(lambda r: f"{r.brier:.3f} ({r.brier_boot_lo:.3f}-{r.brier_boot_hi:.3f})", axis=1)
    disp["Intercept (95% boot)"] = disp.apply(
        lambda r: f"{r.calibration_intercept:+.3f} ({r.calibration_intercept_boot_lo:+.3f} to {r.calibration_intercept_boot_hi:+.3f})", axis=1)
    disp["Slope (95% boot)"] = disp.apply(
        lambda r: f"{r.calibration_slope:.3f} ({r.calibration_slope_boot_lo:.3f}-{r.calibration_slope_boot_hi:.3f})", axis=1)
    print(disp[["mask_variant", "split", "n_total", "prevalence", "mean_predicted_risk",
                "Brier (95% boot)", "Intercept (95% boot)", "Slope (95% boot)"]].to_string(index=False))

    # ---------------- figures ----------------
    for split in FIG_SPLITS:
        fig, (ax, axh) = plt.subplots(
            2, 1, figsize=(5.3, 5.9), sharex=True,
            gridspec_kw={"height_ratios": [3.4, 1.0], "hspace": 0.10})

        ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), color=C.CB["grey"], lw=1.2, zorder=1,
                label="Perfect calibration")

        colors = {"gt": C.CB["blue"], "model": C.CB["vermillion"]}
        labels = {"gt": "Manual (reference) mask", "model": "Automatic nnU-Net mask"}
        markers = {"gt": "o", "model": "s"}
        hist_bins = np.linspace(0, 1, 21)
        count_notes = []

        for variant in C.VARIANTS:
            y, p = C.get_y_p(ARCH, split, variant)
            cc = C.calibration_curve_quantile(y, p, n_bins=10)
            row = sdf[(sdf.mask_variant == variant) & (sdf.split == split)].iloc[0]

            # Wilson limits are not symmetric about the point estimate; clip away
            # float noise so matplotlib accepts the asymmetric error bars.
            yerr = np.clip(np.vstack([cc.obs_freq - cc.obs_lo_wilson,
                                      cc.obs_hi_wilson - cc.obs_freq]), 0.0, None)
            ax.errorbar(cc.mean_pred, cc.obs_freq, yerr=yerr, fmt=markers[variant],
                        color=colors[variant], ecolor=colors[variant], elinewidth=1.0,
                        capsize=2.5, ms=5.5, mfc="white", mew=1.6, lw=1.6, zorder=3,
                        label=(f"{labels[variant]}: intercept {row.calibration_intercept:+.2f}, "
                               f"slope {row.calibration_slope:.2f}, Brier {row.brier:.3f}"))
            axh.hist(p, bins=hist_bins, color=colors[variant], alpha=0.5,
                     edgecolor=colors[variant], linewidth=0.5, zorder=2)
            count_notes.append(f"{labels[variant]}: {int(cc.n.min())}-{int(cc.n.max())}")

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.05, 1.30)   # headroom so the legend never overlaps the data
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylabel("Observed proportion malignant")
        n_all = int(sdf[(sdf.split == split) & (sdf.mask_variant == "gt")].n_total.iloc[0])
        npos = int(sdf[(sdf.split == split) & (sdf.mask_variant == "gt")].n_malignant.iloc[0])
        nneg = int(sdf[(sdf.split == split) & (sdf.mask_variant == "gt")].n_benign.iloc[0])
        ax.set_title(f"{C.PRETTY_SPLIT[split]}\nDenseNet121, n = {n_all} "
                     f"({npos} malignant / {nneg} benign)", pad=8)
        ax.legend(loc="upper left", fontsize=7.8, handletextpad=0.6, labelspacing=0.6)
        ax.tick_params(labelbottom=False)

        axh.set_xlabel("Predicted probability of malignancy")
        axh.set_ylabel("Cases", fontsize=9)
        axh.margins(y=0.12)
        axh.annotate("Calibration points use 10 quantile bins (cases per bin "
                     + "; ".join(count_notes) + "); error bars are Wilson 95% CI.\n"
                     "Lower panel: distribution of predicted probabilities in 20 fixed-width bins.",
                     xy=(0.0, -0.62), xycoords="axes fraction", ha="left", va="top",
                     fontsize=6.8, color=C.CB["grey"])

        C.save_fig(fig, f"task3_calibration_{split}")
        plt.close(fig)


if __name__ == "__main__":
    main()
