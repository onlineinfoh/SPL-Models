#!/usr/bin/env python3
"""Build Table 2b, the boundary-sensitive segmentation metrics.

Reads:
  results/seg_metrics_per_case.csv   per-case HD95 and ASSD, from seg_metrics_engine.py
  results/table2_paired_tests.csv    paired Wilcoxon tests, from make_table2.py

Writes:
  results/table2b_boundary_metrics.csv
  results/table2b_boundary_metrics.md

    python3 analysis/code/make_table2b.py

HD95 and ASSD are strongly right-skewed, so the median (IQR) is the primary
summary and the mean carries a patient-level bootstrap CI on the same seed as
every other interval in the analysis.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seg_metrics_engine import BOOTSTRAP_SEED, bootstrap_ci  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "analysis" / "results"

MODEL_LABEL = {"nnUNet": "nnU-Net", "DeepLabv3plus": "DeepLabv3+", "UNet": "U-Net"}
COHORT_LABEL = {
    "internal_val": "Development/tuning (Center 1)",
    "external_test1": "External Test 1 (Center 2)",
    "external_test2": "External Test 2 (Center 3)",
    "train": "Training (model fit)",
}
METRIC_LABEL = {"hd95": "HD95", "assd": "ASSD"}


def summarise(values: np.ndarray) -> dict:
    """Median, IQR, bootstrap mean CI, p95 and max over the evaluable cases."""
    finite = values[np.isfinite(values)]
    mean, lo, hi = bootstrap_ci(finite)
    return {
        "n_evaluable": len(finite),
        "n_undefined": len(values) - len(finite),
        "Median": round(float(np.median(finite)), 2),
        "IQR": f"{np.percentile(finite, 25):.2f}-{np.percentile(finite, 75):.2f}",
        "Mean": round(mean, 2),
        "Mean_95CI_bootstrap": f"{lo:.2f}-{hi:.2f}",
        "p95": round(float(np.percentile(finite, 95)), 2),
        "Max": round(float(finite.max()), 2),
    }


def main() -> None:
    per_case = pd.read_csv(RESULTS / "seg_metrics_per_case.csv")

    rows = []
    for cohort, cohort_label in COHORT_LABEL.items():
        for model, model_label in MODEL_LABEL.items():
            sub = per_case[(per_case.cohort == cohort) & (per_case.model == model)]
            if sub.empty:
                continue
            for metric, metric_label in METRIC_LABEL.items():
                rows.append(
                    {"Cohort": cohort_label, "Model": model_label, "Metric": metric_label}
                    | summarise(sub[metric].to_numpy(dtype=float))
                )

    csv_path = RESULTS / "table2b_boundary_metrics.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(rows)} rows)")

    paired = pd.read_csv(RESULTS / "table2_paired_tests.csv")
    paired = paired[paired.metric.isin(METRIC_LABEL)]

    out = [
        "# Table 2b - Boundary-sensitive segmentation metrics",
        "",
        "HD95 = 95th percentile of the symmetric boundary distance set.",
        "ASSD = mean of the symmetric boundary distance set.",
        "Both are in **pixels on the native image grid** (median image ~1.25-1.5 megapixels).",
        "Both are undefined when either mask is empty; `n_undefined` counts those cases "
        "and they are excluded.",
        "Distributions are strongly right-skewed, so **median (IQR) is the primary summary**;",
        f"the mean is given with a 95 percent patient-level bootstrap CI "
        f"(2000 resamples, seed {BOOTSTRAP_SEED}).",
    ]

    for cohort_label in COHORT_LABEL.values():
        out += [
            "",
            f"## {cohort_label}",
            "",
            "| Model | Metric | n | Median (IQR) px | Mean (95% CI) px | p95 px | Undefined |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in (r for r in rows if r["Cohort"] == cohort_label):
            out.append(
                f"| {r['Model']} | {r['Metric']} | {r['n_evaluable']} | "
                f"{r['Median']} ({r['IQR']}) | {r['Mean']} ({r['Mean_95CI_bootstrap']}) | "
                f"{r['p95']} | {r['n_undefined']} |"
            )

    out += [
        "",
        "## Paired comparisons, nnU-Net versus each baseline",
        "",
        "Wilcoxon signed-rank on identical cases. Negative difference favours nnU-Net "
        "(smaller distance is better).",
        "",
        "| Cohort | Metric | Comparison | n paired | nnU-Net | Competitor | Difference | p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in paired.iterrows():
        out.append(
            f"| {r.cohort} | {METRIC_LABEL[r.metric]} | {r.comparison} | {r.n_paired} | "
            f"{r.nnUNet_mean:.2f} | {r.competitor_mean:.2f} | {r.difference:.2f} | "
            f"{r.p_value:.2e} |"
        )

    md_path = RESULTS / "table2b_boundary_metrics.md"
    md_path.write_text("\n".join(out) + "\n")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
