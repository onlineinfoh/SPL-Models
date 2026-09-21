# ==========================================================================
# SUPERSEDED BY THE LOCKED RE-ANALYSIS
# ==========================================================================
#
# Internal-only sweep, but ranked architectures by MEAN accuracy across seeds
# rather than the declared rule. Its output locked_selection_and_external.json
# names a different model from the manuscript. Replaced by
# protocol/run_protocol.py phases C and D.
#
# Retained unmodified as the audit record. Produces no reported result.
# See README.md and docs/REPRODUCE.md for the active pipeline.
# ==========================================================================
#
"""
Multi-seed training sweep for the classification stage.

Every architecture is trained over several seeds. Training and architecture
selection use the internal cohort only: no external data is loaded, evaluated or
logged anywhere in the training loop. The external cohorts are scored once, by
`evaluate_locked_winner()`, after the sweep has finished and the architecture has
been fixed. Software versions and seeds are recorded alongside the results.

Model, preprocessing, augmentation and optimiser are imported unchanged from
`binary_classification/train.py`; only the training loop is rewritten here.

Usage
-----
    ~/venvs/prism/bin/python analysis/code/train_locked.py --seeds 67 1234 2025
    ~/venvs/prism/bin/python analysis/code/train_locked.py --evaluate-winner
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "binary_classification"))

# Preprocessing, dataset, architectures and evaluation come from the original
# pipeline unchanged.
from train import (  # type: ignore  # noqa: E402
    LABEL_FILES, TRAIN_IMG_DIR, TRAIN_MASK_DIR, INT_IMG_DIR, INT_MASK_DIR,
    EXT1_IMG_DIR, EXT1_MASK_DIR, EXT2_IMG_DIR, EXT2_MASK_DIR,
    LABEL_SMOOTH, build_model, eval_model, load_labels,
    normalize_case_prefix,
)

# RAM-cached loader, verified bit-identical to train.build_loader by
# verify_cached_dataset.py. Used only for speed.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cached_dataset import build_cached_loader as build_loader  # type: ignore  # noqa: E402

OUT_ROOT = REPO / "analysis" / "runs_locked"
RESULTS = REPO / "analysis" / "results"
LOGS = REPO / "analysis" / "logs"
for _d in (OUT_ROOT, RESULTS, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

ARCHES = [
    "inception_v3", "vgg19",
    "resnet18", "resnet50", "resnet101",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
    "efficientnet_b3", "efficientnet_b4", "efficientnet_b5",
    "densenet121", "densenet201",
]


def environment_record() -> dict:
    import sklearn
    import torchvision
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }


def train_one(arch: str, seed: int, epochs: int, batch_size: int, lr: float,
              patience: int, device: torch.device) -> dict:
    """
    Train one architecture with one seed.

    Only the training and internal-validation cohorts are loaded. The checkpoint
    kept is the one with the highest internal-validation accuracy.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    lbl_train = load_labels(LABEL_FILES["train"])
    lbl_int = load_labels(LABEL_FILES["internal"])
    train_map = {
        normalize_case_prefix(p.name): lbl_train[normalize_case_prefix(p.name)]
        for p in TRAIN_IMG_DIR.glob("*.nii.gz")
        if normalize_case_prefix(p.name) in lbl_train
    }

    train_loader, _ = build_loader(TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_map,
                                   batch_size, is_train=True)
    val_loader, _ = build_loader(INT_IMG_DIR, INT_MASK_DIR, lbl_int,
                                 batch_size, is_train=False)

    model = build_model(arch).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                                     eta_min=1e-6)

    run_dir = OUT_ROOT / f"{arch}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best.pth"
    log_path = run_dir / "train_log.txt"

    best_val_acc = -np.inf
    best_val_auc = np.nan
    best_epoch = None
    patience_ctr = 0
    history: dict[str, list[float]] = defaultdict(list)

    with log_path.open("w") as logf:
        logf.write(f"=== {arch} seed={seed} (internal cohort only) ===\n")
        for epoch in range(1, epochs + 1):
            model.train()
            for x, y, m, _ in train_loader:
                x, y, m = x.to(device), y.to(device), m.to(device)
                optimizer.zero_grad()
                logits = model(x)
                y_smooth = y * (1 - LABEL_SMOOTH) + 0.5 * LABEL_SMOOTH
                lesion_weight = (m.mean(dim=(1, 2)) * 0.9 + 0.1).clamp(min=0.1)
                loss = (criterion(logits, y_smooth) * lesion_weight).mean()
                loss.backward()
                optimizer.step()
            scheduler.step()

            val_metrics = eval_model(model, val_loader, device)
            history["val_acc"].append(float(val_metrics["acc"]))
            history["val_auc"].append(float(val_metrics["auc"]))

            line = (f"[{arch} seed{seed}] epoch {epoch:02d} "
                    f"val_acc={val_metrics['acc']:.4f} val_auc={val_metrics['auc']:.4f}")
            logf.write(line + "\n")
            logf.flush()

            val_acc = val_metrics["acc"]
            if not np.isnan(val_acc) and val_acc > best_val_acc + 1e-6:
                best_val_acc = val_acc
                best_val_auc = float(val_metrics["auc"])
                best_epoch = epoch
                patience_ctr = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    logf.write(f"early stop at epoch {epoch}\n")
                    break

        logf.write(f"BEST epoch={best_epoch} val_acc={best_val_acc:.4f} "
                   f"val_auc={best_val_auc:.4f}\n")

    return {
        "arch": arch, "seed": seed,
        "best_epoch": best_epoch,
        "best_val_acc": float(best_val_acc),
        "best_val_auc": float(best_val_auc),
        "epochs_run": len(history["val_acc"]),
        "ckpt": str(ckpt_path.relative_to(REPO)),
    }


def evaluate_locked_winner(winner_arch: str, winner_seed: int,
                           device: torch.device) -> dict:
    """
    Score the selected checkpoint on the internal and both external cohorts.

    Per-case probabilities are written to results/ as CSV, one file per cohort.
    """
    lbl_int = load_labels(LABEL_FILES["internal"])
    lbl_e1 = load_labels(LABEL_FILES["external1"])
    lbl_e2 = load_labels(LABEL_FILES["external2"])

    model = build_model(winner_arch).to(device)
    ckpt = OUT_ROOT / f"{winner_arch}_seed{winner_seed}" / "best.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device))

    out = {}
    for name, (img_d, mask_d, lbl) in {
        "internal_val": (INT_IMG_DIR, INT_MASK_DIR, lbl_int),
        "external_test1": (EXT1_IMG_DIR, EXT1_MASK_DIR, lbl_e1),
        "external_test2": (EXT2_IMG_DIR, EXT2_MASK_DIR, lbl_e2),
    }.items():
        loader, _ = build_loader(img_d, mask_d, lbl, 16, is_train=False)
        m, (y, p) = eval_model(model, loader, device, return_arrays=True)
        out[name] = {k: (None if v is None or (isinstance(v, float) and np.isnan(v))
                         else float(v)) for k, v in m.items()}
        np.savetxt(RESULTS / f"locked_{winner_arch}_seed{winner_seed}_{name}_probs.csv",
                   np.column_stack([y, p]), delimiter=",",
                   header="label,prob_malignant", comments="")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--arches", nargs="+", default=ARCHES)
    ap.add_argument("--seeds", type=int, nargs="+", default=[67, 1234, 2025])
    ap.add_argument("--evaluate-winner", action="store_true",
                    help="skip training; evaluate the already-selected winner")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = environment_record()
    print(json.dumps(env, indent=2))

    sweep_path = RESULTS / "locked_sweep_results.json"

    if not args.evaluate_winner:
        rows = []
        if sweep_path.exists():
            rows = json.loads(sweep_path.read_text()).get("runs", [])
        done = {(r["arch"], r["seed"]) for r in rows}

        for arch in args.arches:
            for seed in args.seeds:
                if (arch, seed) in done:
                    print(f"skip {arch} seed{seed} (done)")
                    continue
                t0 = datetime.now(timezone.utc)
                print(f"--- training {arch} seed{seed} ---", flush=True)
                try:
                    r = train_one(arch, seed, args.epochs, args.batch_size,
                                  args.lr, args.patience, device)
                except Exception as e:  # keep the sweep alive
                    print(f"!! {arch} seed{seed} FAILED: {e}", flush=True)
                    r = {"arch": arch, "seed": seed, "error": str(e),
                         "best_val_acc": float("nan"), "best_val_auc": float("nan")}
                r["minutes"] = (datetime.now(timezone.utc) - t0).total_seconds() / 60
                rows.append(r)
                print(f"    -> val_acc={r.get('best_val_acc'):.4f} "
                      f"val_auc={r.get('best_val_auc'):.4f} "
                      f"({r['minutes']:.1f} min)", flush=True)
                sweep_path.write_text(json.dumps(
                    {"environment": env, "runs": rows}, indent=2))
        print(f"\nsweep written to {sweep_path}")

    # ---- architecture selection on internal validation --------------------
    rows = json.loads(sweep_path.read_text())["runs"]
    valid = [r for r in rows if np.isfinite(r.get("best_val_acc", np.nan))]
    by_arch: dict[str, list[float]] = defaultdict(list)
    for r in valid:
        by_arch[r["arch"]].append(r["best_val_acc"])
    arch_mean = sorted(((a, float(np.mean(v)), float(np.std(v)), len(v))
                        for a, v in by_arch.items()),
                       key=lambda t: -t[1])

    print("\n=== architecture ranking by MEAN internal-validation accuracy "
          "(external data never seen) ===")
    for a, m, s, n in arch_mean:
        print(f"  {a:18s} mean_val_acc={m:.4f} sd={s:.4f} n_seeds={n}")

    best_arch = arch_mean[0][0]
    best_run = max((r for r in valid if r["arch"] == best_arch),
                   key=lambda r: r["best_val_acc"])
    print(f"\nSELECTED (internal val only): {best_arch} seed={best_run['seed']}")

    ext = evaluate_locked_winner(best_arch, best_run["seed"], device)
    print("\n=== external evaluation, performed once, after selection ===")
    print(json.dumps(ext, indent=2))

    (RESULTS / "locked_selection_and_external.json").write_text(json.dumps({
        "environment": env,
        "arch_ranking_internal_only": [
            {"arch": a, "mean_val_acc": m, "sd": s, "n_seeds": n}
            for a, m, s, n in arch_mean],
        "selected_arch": best_arch,
        "selected_seed": best_run["seed"],
        "external_evaluation_after_lock": ext,
    }, indent=2))
    print(f"\nwrote {RESULTS/'locked_selection_and_external.json'}")


if __name__ == "__main__":
    main()
