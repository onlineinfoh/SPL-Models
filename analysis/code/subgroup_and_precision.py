"""
DenseNet121 performance by lesion size, and achieved precision of each cohort.

Subgroups are lesion-size tertiles derived from the ground-truth area fraction
in results/seg_metrics_per_case.csv, so no clinical metadata is needed. Tertiles
are computed within each cohort, to avoid confounding by cohort-level
differences in lesion size.

Precision is reported as the achieved half-width of the 95 percent interval,
together with the sample size that a target half-width would have required.
Retrospective power is not reported; it is a function of the observed p-value.

Usage:
    ~/venvs/prism/bin/python analysis/code/subgroup_and_precision.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
import os, sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

# Honours SPL_PRED_DIR / SPL_ARCH / SPL_RESULTS like the other analysis scripts.
PRED = C.PRED_DIR / C.ARCH
RES = C.RESULTS   # SPL_RESULTS env var; defaults to analysis/results

# Lesion-size tertiles come from the GROUND-TRUTH area fraction, which does not
# depend on which segmentation model produced the predicted masks. This input is
# therefore read from the segmentation metrics location rather than from the
# active results directory, so the classification analysis can be regenerated
# independently of the segmentation rebuild.
SEG_METRICS = Path(os.environ.get(
    "SPL_SEG_METRICS", REPO / "analysis" / "results" / "seg_metrics_per_case.csv"))

COHORTS = {"internal_val": "Development/tuning (Center 1)",
           "external_test1": "External Test 1 (Center 2)",
           "external_test2": "External Test 2 (Center 3)"}
THR = 0.5000  # automatic-mask operating point, selected on the tuning cohort


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def boot_auc_ci(y, p, n_boot=2000, seed=20260904):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(pos) < 2 or len(neg) < 2:
        return float("nan"), float("nan")
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), True),
                              rng.choice(neg, len(neg), True)])
        if len(np.unique(y[idx])) > 1:
            out.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> None:
    per_case = pd.read_csv(SEG_METRICS)
    area = (per_case[per_case.model == "nnUNet"]
            .set_index(["cohort", "case_id"])["gt_area_frac"])

    sub_rows, prec_rows = [], []

    for ck, cl in COHORTS.items():
        df = pd.read_csv(PRED / f"{ck}_model_seed67_probs.txt")
        df["area_frac"] = [area.get((ck, c), np.nan) for c in df.case_id]
        df = df.dropna(subset=["area_frac"])
        y = df.label.to_numpy()
        p = df.prob_malignant.to_numpy()

        # ---------- precision of the whole cohort ----------
        n_mal, n_ben = int((y == 1).sum()), int((y == 0).sum())
        auc = roc_auc_score(y, p)
        lo, hi = boot_auc_ci(y, p)
        pred = (p >= THR).astype(int)
        sens, s_lo, s_hi = wilson(int(((pred == 1) & (y == 1)).sum()), n_mal)
        spec, p_lo, p_hi = wilson(int(((pred == 0) & (y == 0)).sum()), n_ben)

        def need(phat, target_hw):
            # n for a Wald half-width `target_hw` at the observed proportion
            return int(np.ceil(1.96 ** 2 * phat * (1 - phat) / target_hw ** 2))

        prec_rows.append({
            "cohort": cl, "n": len(df), "n_malignant": n_mal, "n_benign": n_ben,
            "AUC": round(auc, 4), "AUC_95CI": f"{lo:.3f}-{hi:.3f}",
            "AUC_half_width": round((hi - lo) / 2, 4),
            "sensitivity": round(sens, 4), "sens_95CI": f"{s_lo:.3f}-{s_hi:.3f}",
            "sens_half_width": round((s_hi - s_lo) / 2, 4),
            "specificity": round(spec, 4), "spec_95CI": f"{p_lo:.3f}-{p_hi:.3f}",
            "spec_half_width": round((p_hi - p_lo) / 2, 4),
            "n_malignant_needed_for_sens_hw_0.05": need(sens, 0.05),
            "n_benign_needed_for_spec_hw_0.05": need(spec, 0.05),
            "n_benign_needed_for_spec_hw_0.10": need(spec, 0.10),
        })

        # ---------- lesion-size subgroups, tertiles within cohort ----------
        q1, q2 = df.area_frac.quantile([1 / 3, 2 / 3])
        df["size_group"] = np.where(df.area_frac <= q1, "small (T1)",
                            np.where(df.area_frac <= q2, "medium (T2)", "large (T3)"))
        for g in ["small (T1)", "medium (T2)", "large (T3)"]:
            s = df[df.size_group == g]
            ys, ps = s.label.to_numpy(), s.prob_malignant.to_numpy()
            nm, nb = int((ys == 1).sum()), int((ys == 0).sum())
            a = roc_auc_score(ys, ps) if nm > 0 and nb > 0 else float("nan")
            alo, ahi = boot_auc_ci(ys, ps) if nm > 1 and nb > 1 else (np.nan, np.nan)
            prd = (ps >= THR).astype(int)
            se, _, _ = wilson(int(((prd == 1) & (ys == 1)).sum()), nm)
            sp, _, _ = wilson(int(((prd == 0) & (ys == 0)).sum()), nb)
            sub_rows.append({
                "cohort": cl, "subgroup": g, "n": len(s),
                "lesion_area_frac_range": f"{s.area_frac.min():.4f}-{s.area_frac.max():.4f}",
                "n_malignant": nm, "n_benign": nb,
                "prevalence": round(nm / len(s), 3),
                "AUC": round(a, 4) if np.isfinite(a) else None,
                "AUC_95CI": f"{alo:.3f}-{ahi:.3f}" if np.isfinite(alo) else "n/a",
                "sensitivity": round(se, 4) if nm else None,
                "specificity": round(sp, 4) if nb else None,
            })

    sub = pd.DataFrame(sub_rows)
    prec = pd.DataFrame(prec_rows)
    sub.to_csv(RES / "subgroup_by_lesion_size.csv", index=False)
    prec.to_csv(RES / "external_cohort_precision.csv", index=False)

    print("=" * 100)
    print("SUBGROUP PERFORMANCE BY LESION SIZE (tertiles within cohort), "
          "DenseNet121, automatic masks")
    print("=" * 100)
    print(sub.to_string(index=False))
    print()
    print("=" * 100)
    print("ACHIEVED PRECISION OF THE EXTERNAL COHORTS")
    print("=" * 100)
    print(prec[["cohort", "n", "n_malignant", "n_benign", "AUC_95CI",
                "AUC_half_width", "sens_half_width", "spec_half_width",
                "n_benign_needed_for_spec_hw_0.05",
                "n_benign_needed_for_spec_hw_0.10"]].to_string(index=False))

    (RES / "subgroup_and_precision.json").write_text(json.dumps(
        {"subgroups": sub_rows, "precision": prec_rows}, indent=2))
    print(f"\nwrote {RES/'subgroup_by_lesion_size.csv'}")
    print(f"wrote {RES/'external_cohort_precision.csv'}")


if __name__ == "__main__":
    main()
