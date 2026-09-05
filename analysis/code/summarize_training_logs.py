"""
Parse the training logs and report which epoch each selection criterion picks.

For every architecture in binary_classification/runs the per-epoch series is
recovered from the log and compared:
  - the epoch actually retained
  - the epoch maximising internal-validation accuracy
  - the epoch maximising External Test 1 / Test 2 AUC
  - the difference in external AUC between the retained epoch and the best epoch

The logs under runs_locked/ are also scanned for external metric tokens, and the
distribution of retained epochs across those runs is summarised.

Usage:
    ~/venvs/prism/bin/python analysis/code/summarize_training_logs.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ORIG = REPO / "binary_classification" / "runs"
LOCKED = REPO / "analysis" / "runs_locked"
OUT = REPO / "analysis" / "results"

HDR = re.compile(
    r"\[(?P<arch>\S+) Seed (?P<seed>\d+)\] Epoch (?P<epoch>\d+) "
    r"train_auc=(?P<train_auc>[\d.nan]+) val_auc=(?P<val_auc>[\d.nan]+) "
    r"ext1_auc=(?P<ext1>[\d.nan]+) ext2_auc=(?P<ext2>[\d.nan]+)")
VAL = re.compile(r"^\s+val: acc=(?P<acc>[\d.nan]+)")
BEST = re.compile(r"best_epoch=(?P<epoch>\d+) val_acc=(?P<acc>[\d.]+)")


def _f(x: str) -> float:
    try:
        return float(x)
    except ValueError:
        return float("nan")


def parse_original(path: Path) -> tuple[pd.DataFrame, int | None]:
    rows, best_epoch, pending = [], None, None
    for line in path.read_text().splitlines():
        m = HDR.search(line)
        if m:
            pending = {"epoch": int(m["epoch"]), "val_auc": _f(m["val_auc"]),
                       "ext1_auc": _f(m["ext1"]), "ext2_auc": _f(m["ext2"])}
            continue
        v = VAL.match(line)
        if v and pending is not None:
            pending["val_acc"] = _f(v["acc"])
            rows.append(pending)
            pending = None
            continue
        b = BEST.search(line)
        if b:
            best_epoch = int(b["epoch"])
    return pd.DataFrame(rows), best_epoch


def main() -> None:
    records = []
    for d in sorted(ORIG.iterdir()):
        log = d / "train_log.txt"
        if not log.exists():
            continue
        df, best_epoch = parse_original(log)
        if df.empty or best_epoch is None:
            continue

        # first epoch attaining the maximum internal-validation accuracy
        sel_val = int(df.loc[df.val_acc.idxmax(), "epoch"])
        arg_e1 = int(df.loc[df.ext1_auc.idxmax(), "epoch"])
        arg_e2 = int(df.loc[df.ext2_auc.idxmax(), "epoch"])
        row_sel = df[df.epoch == best_epoch].iloc[0]

        records.append({
            "architecture": d.name,
            "epochs_run": len(df),
            "epoch_retained": best_epoch,
            "epoch_argmax_internal_val_acc": sel_val,
            "retained_matches_internal_rule": best_epoch == sel_val,
            "epoch_argmax_ext1_auc": arg_e1,
            "epoch_argmax_ext2_auc": arg_e2,
            "retained_is_argmax_ext1": best_epoch == arg_e1,
            "retained_is_argmax_ext2": best_epoch == arg_e2,
            "ext1_auc_at_retained": round(float(row_sel.ext1_auc), 4),
            "ext1_auc_best_available": round(float(df.ext1_auc.max()), 4),
            "ext1_auc_forgone": round(float(df.ext1_auc.max() - row_sel.ext1_auc), 4),
            "ext2_auc_at_retained": round(float(row_sel.ext2_auc), 4),
            "ext2_auc_best_available": round(float(df.ext2_auc.max()), 4),
            "ext2_auc_forgone": round(float(df.ext2_auc.max() - row_sel.ext2_auc), 4),
        })

    t = pd.DataFrame(records)
    t.to_csv(OUT / "checkpoint_selection_summary.csv", index=False)

    n = len(t)
    n_int = int(t.retained_matches_internal_rule.sum())
    n_e1 = int(t.retained_is_argmax_ext1.sum())
    n_e2 = int(t.retained_is_argmax_ext2.sum())

    print("=" * 78)
    print("SELECTION AUDIT of the original training logs")
    print("=" * 78)
    print(f"architectures audited                                : {n}")
    print(f"retained epoch == argmax internal-validation accuracy: {n_int}/{n}")
    print(f"retained epoch == argmax External Test 1 AUC         : {n_e1}/{n}")
    print(f"retained epoch == argmax External Test 2 AUC         : {n_e2}/{n}")
    print(f"mean External Test 1 AUC forgone by following the internal rule: "
          f"{t.ext1_auc_forgone.mean():.4f}")
    print(f"mean External Test 2 AUC forgone by following the internal rule: "
          f"{t.ext2_auc_forgone.mean():.4f}")
    print()
    print(t[["architecture", "epoch_retained", "epoch_argmax_internal_val_acc",
             "epoch_argmax_ext1_auc", "epoch_argmax_ext2_auc",
             "ext1_auc_forgone", "ext2_auc_forgone"]].to_string(index=False))

    # ---- scan the runs_locked logs for external metrics ---------------------
    # The per-run banner line itself contains the word "external", so searching
    # for that word alone gives a false hit. Match metric tokens instead and skip
    # the banner.
    banned = ("ext1_auc", "ext2_auc", "ext1:", "ext2:", "external_test")
    scanned, hits, best_epochs = 0, [], []
    for log in sorted(LOCKED.glob("*/train_log.txt")):
        scanned += 1
        for line in log.read_text().splitlines():
            if line.startswith("=== ") and "internal cohort only" in line:
                continue
            low = line.lower()
            for b in banned:
                if b in low:
                    hits.append((log.parent.name, b, line[:80]))
        m = re.search(r"BEST epoch=(\d+)", log.read_text())
        if m:
            best_epochs.append(int(m[1]))

    print()
    print("=" * 78)
    print(f"sweep logs scanned: {scanned}")
    print(f"external metric tokens found (ext1_auc/ext2_auc/ext1:/ext2:/external_test): {len(hits)}")
    print("VERIFIED: no external cohort metric appears in any sweep log."
          if not hits else f"UNEXPECTED: {hits[:10]}")

    be = np.array(best_epochs)
    print()
    print("Early-stopping stability across the sweep runs:")
    print(f"  retained epoch: median {int(np.median(be))}, min {be.min()}, max {be.max()}")
    print(f"  runs retaining epoch 1 or 2: {int((be <= 2).sum())}/{len(be)}")
    print("=" * 78)

    (OUT / "checkpoint_selection_summary.json").write_text(json.dumps({
        "n_architectures": n,
        "retained_equals_internal_argmax": n_int,
        "retained_equals_ext1_argmax": n_e1,
        "retained_equals_ext2_argmax": n_e2,
        "mean_ext1_auc_forgone": float(t.ext1_auc_forgone.mean()),
        "mean_ext2_auc_forgone": float(t.ext2_auc_forgone.mean()),
        "sweep_logs_scanned": scanned,
        "external_metric_mentions": len(hits),
    }, indent=2))
    print(f"\nwrote {OUT/'checkpoint_selection_summary.csv'}")


if __name__ == "__main__":
    main()
