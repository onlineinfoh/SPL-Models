#!/usr/bin/env python3
"""Audit all existing original predictions and both saved sweeps; never train.

Accuracy uses each original model's maximum-internal-accuracy threshold, applied
unchanged externally. AUC uses the same saved probabilities. These are separate
retrospective rankings, not an attempt to infer a historical selection timeline.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

import common as C

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "selection_audit"


def main():
    # Pin historical inputs regardless of environment overrides.
    C.PRED_DIR = REPO / "binary_classification" / "predictions_tight"
    rows, sources = [], set()
    for variant in C.VARIANTS:
        for arch in C.ARCHS:
            yv, pv = C.get_y_p(arch, "internal_val", variant)
            threshold = C.threshold_max_accuracy(yv, pv)
            for split in ("internal_val", "external_test1", "external_test2"):
                y, p = C.get_y_p(arch, split, variant)
                sources.add(C.pred_path(arch, split, variant))
                for metric, value in (
                    ("accuracy", float(((p >= threshold) == y).mean())),
                    ("auc", float(roc_auc_score(y, p))),
                ):
                    rows.append(dict(experiment="original_224px", seed=67,
                                     variant=variant, split=split, arch=arch,
                                     metric=metric, value=value, threshold=threshold))

    for experiment, filename, fields in (
        ("earlier_accuracy_sweep", "analysis/results/locked_sweep_results.json",
         {"accuracy": "best_val_acc", "auc": "best_val_auc"}),
        ("revision_auc_sweep", "protocol/results/internal_sweep.json",
         {"accuracy": "internal_val_accuracy", "auc": "internal_val_auc"}),
    ):
        path = REPO / filename
        sources.add(path)
        runs = json.loads(path.read_text())["runs"]
        assert len(runs) == 39
        assert len({(r["arch"], r["seed"]) for r in runs}) == 39
        for run in runs:
            for metric, field in fields.items():
                rows.append(dict(experiment=experiment, seed=run["seed"],
                                 variant="gt", split="internal_val", arch=run["arch"],
                                 metric=metric, value=run[field],
                                 threshold=(0.5 if experiment == "earlier_accuracy_sweep"
                                            else run["internal_val_threshold_max_acc"])))

    frame = pd.DataFrame(rows)
    keys = ["experiment", "seed", "variant", "split", "metric"]
    groups = frame.groupby(keys)
    assert (groups.size() == 13).all(), "incomplete architecture comparison"
    frame["rank"] = groups.value.rank(ascending=False, method="min").astype(int)
    frame["n_compared"] = groups.value.transform("size")
    frame["n_tied_at_value"] = frame.groupby(keys + ["value"]).value.transform("size")
    frame = frame.sort_values(keys + ["rank", "arch"])
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "all_rankings.csv", index=False)
    dense = frame[frame.arch == "densenet121"]
    dense.to_csv(OUT / "densenet121_rankings.csv", index=False)
    (OUT / "sources.json").write_text(json.dumps({
        "scope": "All 13 original models and both recorded 39-run sweeps; no new runs",
        "limitations": [
            "Sweep accuracy definitions differ; compare ranks only within each experiment.",
            "Sweep AUC is measured at each run's retained checkpoint, not a new checkpoint search.",
            "No full all-architecture external comparison is stored for either 39-run sweep.",
            "Only seeds 67, 1234, 2025 are represented in the recorded sweeps.",
            "Original prediction files use 224 px inference, not corrected 300 px inference.",
            "An observed ranking does not establish when or how the model was originally selected.",
        ],
        "sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(sources)},
    }, indent=2) + "\n")
    print(dense[["experiment", "seed", "variant", "split", "metric", "value", "rank"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
