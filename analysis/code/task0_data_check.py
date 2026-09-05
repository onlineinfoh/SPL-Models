#!/usr/bin/env python3
"""
Task 0: integrity checks on the per-case probability files.

Checks cohort sizes and malignant/benign counts against the expected values,
looks for duplicate case ids and probabilities outside [0, 1], checks that
prob_malignant + prob_benign == 1, and checks the DeLong AUC implementation
against sklearn's roc_auc_score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import common as C


def main():
    C.ensure_dirs()
    rows = []
    problems = []
    for arch in C.ARCHS:
        for variant in C.VARIANTS:
            for split in C.SPLITS:
                p = C.pred_path(arch, split, variant)
                if not p.exists():
                    problems.append(f"MISSING FILE: {p}")
                    continue
                df = C.load_probs(arch, split, variant)
                n = len(df)
                npos = int((df.label == 1).sum())
                nneg = int((df.label == 0).sum())
                exp_n = C.EXPECTED_N[split]
                exp_pn = C.EXPECTED_POSNEG[split]
                ok_n = (n == exp_n)
                ok_pn = True if exp_pn is None else ((npos, nneg) == exp_pn)
                dup = int(df.case_id.duplicated().sum())
                sum_dev = float(np.max(np.abs(df.prob_malignant + df.prob_benign - 1.0)))
                oob = int(((df.prob_malignant < 0) | (df.prob_malignant > 1)).sum())
                if not ok_n:
                    problems.append(f"N MISMATCH {arch}/{variant}/{split}: got {n}, expected {exp_n}")
                if not ok_pn:
                    problems.append(f"CLASS MISMATCH {arch}/{variant}/{split}: got {npos}/{nneg}, expected {exp_pn}")
                if dup:
                    problems.append(f"DUPLICATE CASE IDS {arch}/{variant}/{split}: {dup}")
                if oob:
                    problems.append(f"PROB OUT OF RANGE {arch}/{variant}/{split}: {oob}")
                if sum_dev > 2e-6:
                    problems.append(f"PROB SUM != 1 {arch}/{variant}/{split}: max dev {sum_dev:.2e}")
                rows.append({
                    "arch": arch, "mask_variant": variant, "split": split,
                    "n": n, "n_malignant": npos, "n_benign": nneg,
                    "expected_n": exp_n,
                    "expected_malignant": None if exp_pn is None else exp_pn[0],
                    "expected_benign": None if exp_pn is None else exp_pn[1],
                    "n_matches_expected": ok_n,
                    "class_counts_match_expected": ok_pn,
                    "duplicate_case_ids": dup,
                    "max_abs_prob_sum_deviation": sum_dev,
                    "prevalence_malignant": npos / n if n else np.nan,
                })
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS / "task0_data_integrity_check.csv", index=False)

    # DeLong vs sklearn AUC, over every densenet121 cohort
    checks = []
    for variant in C.VARIANTS:
        for split in C.SPLITS:
            y, p = C.get_y_p("densenet121", split, variant)
            sk = roc_auc_score(y, p)
            dl = C.delong_auc_ci(y, p)["auc"]
            checks.append({"mask_variant": variant, "split": split,
                           "auc_sklearn": sk, "auc_delong_impl": dl,
                           "abs_diff": abs(sk - dl)})
    chk = pd.DataFrame(checks)
    chk.to_csv(C.RESULTS / "task0_delong_implementation_check.csv", index=False)
    worst = chk["abs_diff"].max()
    if worst > 1e-10:
        problems.append(f"DeLong AUC disagrees with sklearn by up to {worst:.3e}")

    print("=== Task 0: data integrity ===")
    print(out[out.arch == "densenet121"][
        ["mask_variant", "split", "n", "n_malignant", "n_benign",
         "n_matches_expected", "class_counts_match_expected"]].to_string(index=False))
    print(f"\nAll {len(out)} arch x variant x split files checked.")
    print(f"DeLong vs sklearn max |diff| = {worst:.3e}")
    if problems:
        print("\n!!! PROBLEMS FOUND !!!")
        for pr in problems:
            print("  -", pr)
    else:
        print("\nNo problems found: all cohort sizes and class counts match the pre-specified values.")

    with open(C.RESULTS / "task0_problems.txt", "w") as fh:
        if problems:
            fh.write("\n".join(problems) + "\n")
        else:
            fh.write("NONE - all cohort sizes, class counts, and integrity checks passed.\n")


if __name__ == "__main__":
    main()
