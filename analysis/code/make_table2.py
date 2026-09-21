"""
Build the segmentation performance table from seg_metrics_engine.py output.

Reads results/seg_metrics_summary.json and results/seg_metrics_per_case.csv.

Outputs
-------
  results/table2_corrected.csv          the table, one row per cohort and model
  results/table2_corrected.md           same table as markdown
  results/table2_paired_tests.csv       paired Wilcoxon, nnU-Net vs each model
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
import os
RES = Path(os.environ.get("SPL_RESULTS", REPO / "analysis" / "results"))   # avoids clobbering the superseded Table 2

MODELS = ["nnUNet", "DeepLabv3plus", "UNet"]
MODEL_LABEL = {"nnUNet": "nnU-Net", "DeepLabv3plus": "DeepLabv3+", "UNet": "U-Net"}
COHORTS = ["train", "internal_val", "external_test1", "external_test2"]
COHORT_LABEL = {
    "train": "Training (model fit)",
    "internal_val": "Development/tuning (Center 1)",
    "external_test1": "External Test 1 (Center 2)",
    "external_test2": "External Test 2 (Center 3)",
}



def fmt(mean: float, lo: float, hi: float, nd: int = 3) -> str:
    if not np.isfinite(mean):
        return "n/a"
    return f"{mean:.{nd}f} ({lo:.{nd}f}-{hi:.{nd}f})"


def main() -> None:
    summary = json.loads((RES / "seg_metrics_summary.json").read_text())
    per_case = pd.read_csv(RES / "seg_metrics_per_case.csv")

    # ---------------- table rows -----------------------------------------
    rows = []
    for cohort in COHORTS:
        for model in MODELS:
            s = summary[model][cohort]
            g = per_case[(per_case.model == model) & (per_case.cohort == cohort)]
            rows.append({
                "Cohort": COHORT_LABEL[cohort],
                "Model": MODEL_LABEL[model],
                "n": s["n_cases"],
                "Dice": fmt(s["dice"]["mean"], s["dice"]["boot_lo"], s["dice"]["boot_hi"]),
                "IoU": fmt(s["iou"]["mean"], s["iou"]["boot_lo"], s["iou"]["boot_hi"]),
                "Precision": fmt(s["precision"]["mean"], s["precision"]["boot_lo"], s["precision"]["boot_hi"]),
                "Recall": fmt(s["recall"]["mean"], s["recall"]["boot_lo"], s["recall"]["boot_hi"]),
                "FPR": f"{s['fpr']['mean']:.5f} ({s['fpr']['boot_lo']:.5f}-{s['fpr']['boot_hi']:.5f})",
                "HD95_median_IQR_px": f"{g.hd95.median():.1f} ({g.hd95.quantile(.25):.1f}-{g.hd95.quantile(.75):.1f})",
                "HD95_mean_px": f"{s['hd95']['mean']:.1f} ({s['hd95']['boot_lo']:.1f}-{s['hd95']['boot_hi']:.1f})",
                "ASSD_median_IQR_px": f"{g.assd.median():.1f} ({g.assd.quantile(.25):.1f}-{g.assd.quantile(.75):.1f})",
                "Empty_predictions": s["n_pred_empty"],
                "Mean_lesion_area_fraction": f"{s['mean_gt_area_frac']:.4f}",
            })
    df = pd.DataFrame(rows)
    df.to_csv(RES / "table2_corrected.csv", index=False)

    # ---------------- paired model comparisons ---------------------------
    ptests = []
    for cohort in COHORTS:
        ref = per_case[(per_case.model == "nnUNet") & (per_case.cohort == cohort)].set_index("case_id")
        for model in ("DeepLabv3plus", "UNet"):
            comp = per_case[(per_case.model == model) & (per_case.cohort == cohort)].set_index("case_id")
            common = ref.index.intersection(comp.index)
            for metric in ("dice", "iou", "hd95", "assd"):
                a = ref.loc[common, metric].to_numpy(float)
                b = comp.loc[common, metric].to_numpy(float)
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() < 5:
                    continue
                stat, p = stats.wilcoxon(a[ok], b[ok])
                ptests.append({
                    "cohort": cohort, "metric": metric,
                    "comparison": f"nnU-Net vs {MODEL_LABEL[model]}",
                    "n_paired": int(ok.sum()),
                    "nnUNet_mean": float(a[ok].mean()),
                    "competitor_mean": float(b[ok].mean()),
                    "difference": float(a[ok].mean() - b[ok].mean()),
                    "wilcoxon_stat": float(stat), "p_value": float(p),
                })
    pd.DataFrame(ptests).to_csv(RES / "table2_paired_tests.csv", index=False)

    # ---------------- markdown table -------------------------------------
    lines = [
        "# Table 2 (corrected) - Segmentation performance",
        "",
        "All metrics are computed on the LESION (foreground) class only, at native image",
        "resolution, from the locked predicted masks, using one shared metric engine",
        "(`analysis/code/seg_metrics_engine.py`).",
        "Values are mean (95% patient-level bootstrap percentile CI, 2000 resamples,",
        "seed 20260904), except HD95 and ASSD which are median (IQR) because their",
        "distributions are right-skewed.",
        "",
        "Explicit denominators:",
        "",
        "- Precision = TP / (TP + FP), denominator = all pixels PREDICTED lesion.",
        "- Recall = TP / (TP + FN), denominator = all pixels TRULY lesion.",
        "- FPR = FP / (FP + TN), denominator = all pixels TRULY background.",
        "- HD95 and ASSD are in pixels on the native grid, undefined when either mask is empty.",
        "",
    ]
    for cohort in COHORTS:
        sub = df[df.Cohort == COHORT_LABEL[cohort]]
        lines += [f"## {COHORT_LABEL[cohort]} (n = {sub.iloc[0]['n']})", ""]
        cols = ["Model", "Dice", "IoU", "Precision", "Recall", "FPR",
                "HD95_median_IQR_px", "ASSD_median_IQR_px", "Empty_predictions"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for _, r in sub.iterrows():
            lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        lines.append("")
    (RES / "table2_corrected.md").write_text("\n".join(lines))

    print(df.to_string(index=False))
    print(f"\nwrote {RES/'table2_corrected.csv'}")
    print(f"wrote {RES/'table2_corrected.md'}")
    print(f"wrote {RES/'table2_paired_tests.csv'}")


if __name__ == "__main__":
    main()
