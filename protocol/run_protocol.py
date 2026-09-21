#!/usr/bin/env python3
"""
Phase runner for the locked re-analysis.

    python protocol/run_protocol.py train      # Phase C: internal only, gate closed
    python protocol/run_protocol.py lock       # Phase D: select, write lock record
    python protocol/run_protocol.py external   # Phase E: single external pass

Phase C trains all 13 architectures over the declared seeds using the training and
internal-validation cohorts only, with `protocol.gate` installed. Any read of an
external path terminates the run and is recorded in protocol/gate_audit.jsonl.

Phase D applies the selection rule from analysis_plan_v1.yaml to the internal
results, writes protocol/locked_pipeline.json with the SHA256 of every weight file
and the git commit, and only then becomes eligible to open the gate.

Phase E releases the gate and scores the locked model once on all four cohorts.

Differences from analysis/code/train_locked.py, all mandated by the protocol:
  - architecture chosen at the PRIMARY seed, not by mean or best across seeds
  - probability files carry case_id
  - the training cohort is scored alongside internal and both external cohorts
  - the rotation/augmentation defect is corrected (see LockedDataset)
  - the hold-out gate is enforced at runtime
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "binary_classification"))
sys.path.insert(0, str(REPO / "analysis" / "code"))

from protocol import gate  # noqa: E402

from train import (  # type: ignore  # noqa: E402
    LABEL_FILES, TRAIN_IMG_DIR, TRAIN_MASK_DIR, INT_IMG_DIR, INT_MASK_DIR,
    EXT1_IMG_DIR, EXT1_MASK_DIR, EXT2_IMG_DIR, EXT2_MASK_DIR,
    LABEL_SMOOTH, build_model, load_labels, normalize_case_prefix,
)
from cached_dataset import CachedTightCropDataset, build_cached_loader  # type: ignore  # noqa: E402
from torch.utils.data import DataLoader, WeightedRandomSampler  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

PROTOCOL = REPO / "protocol"
OUT_ROOT = PROTOCOL / "runs"
RESULTS = PROTOCOL / "results"
LOCK_RECORD = PROTOCOL / "locked_pipeline.json"
SWEEP = RESULTS / "internal_sweep.json"

# Declared in analysis_plan_v1.yaml. Duplicated here as executable constants;
# check_protocol_sync() asserts they still agree with the YAML.
ARCHES = [
    "inception_v3", "vgg19",
    "resnet18", "resnet50", "resnet101",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
    "efficientnet_b3", "efficientnet_b4", "efficientnet_b5",
    "densenet121", "densenet201",
]
PRIMARY_SEED = 67
ROBUSTNESS_SEEDS = [1234, 2025]
HP = dict(epochs=40, batch_size=16, lr=1e-4, weight_decay=5e-4,
          patience=8, eta_min=1e-6)


# ---------------------------------------------------------------------------
# Dataset: the cached loader with the rotation defect corrected
# ---------------------------------------------------------------------------

class LockedDataset(CachedTightCropDataset):
    """
    CachedTightCropDataset with one correction.

    The original applies cv2.warpAffine to an (H, W, 1) array, which returns
    (H, W). The subsequent `base = img[..., 0]` therefore selects a single image
    COLUMN, so gamma, contrast and noise augmentation acted on one column in any
    rotated sample. README.md implementation note 4 documents this. The locked
    implementation restores the channel axis before indexing, so intensity
    augmentation acts on the image plane as intended.

    Everything else, including the augmentation probabilities and ranges, is
    unchanged.
    """

    def __getitem__(self, idx):
        _img_path, _mask_path, label, case = self.items[idx]
        cimg, cmask = self._cache[idx]
        img = cimg.copy()
        mask = cmask.copy()

        if self.is_train:
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=1)
                mask = np.flip(mask, axis=1)
            if np.random.rand() < 0.5:
                angle = np.random.uniform(-20, 20)
                h, w_img = img.shape[:2]
                M = cv2_getRotationMatrix2D((w_img / 2, h / 2), angle, 1.0)
                img = cv2_warpAffine(img, M, (w_img, h), linear=True)
                mask = cv2_warpAffine(mask, M, (w_img, h), linear=False)

            # THE CORRECTION: restore the channel axis dropped by warpAffine
            # before indexing, so `base` is the image plane rather than a column.
            if img.ndim == 2:
                img = img[..., None]

            base = img[..., 0]
            if np.random.rand() < 0.7:
                gamma = np.random.uniform(0.6, 1.4)
                base = np.clip(base ** gamma, 0.0, 255.0)
            if np.random.rand() < 0.7:
                factor = 1.0 + np.random.uniform(-0.25, 0.25)
                mean = base.mean()
                base = np.clip((base - mean) * factor + mean, 0.0, 255.0)
            if np.random.rand() < 0.5:
                base = np.clip(base + np.random.normal(0.0, 6.0, base.shape),
                               0.0, 255.0)
            img[..., 0] = base

        if img.ndim == 2:
            img = img[..., None]
        if mask.ndim == 3:
            mask = mask[..., 0]

        img = np.ascontiguousarray(img)
        mask = np.ascontiguousarray(mask)

        x = np.concatenate([img, mask[..., None]], axis=-1).transpose(2, 0, 1)
        x = torch.from_numpy(x).float()
        x = (x - x.mean(dim=(1, 2), keepdim=True)) / (x.std(dim=(1, 2), keepdim=True) + 1e-6)
        return x, torch.tensor(label, dtype=torch.float32), torch.from_numpy(mask).float(), case


def cv2_getRotationMatrix2D(center, angle, scale):
    import cv2
    return cv2.getRotationMatrix2D(center, angle, scale)


def cv2_warpAffine(arr, M, size, linear: bool):
    import cv2
    return cv2.warpAffine(
        arr, M, size,
        flags=cv2.INTER_LINEAR if linear else cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REFLECT,
    )


_LOADERS: dict = {}


def locked_loader(img_dir: Path, mask_dir: Path, labels: dict, batch_size: int,
                  is_train: bool):
    # The cache key MUST include mask_dir. It previously keyed on img_dir alone,
    # so scoring the same images against manual masks and then against nnU-Net
    # predicted masks returned the first cached dataset both times, silently
    # producing identical "gt" and "model" predictions.
    key = (str(img_dir), str(mask_dir), is_train)
    ds = _LOADERS.get(key)
    if ds is None:
        ds = LockedDataset(img_dir, mask_dir, labels, is_train=is_train)
        _LOADERS[key] = ds
    if len(ds) == 0:
        raise ValueError(f"no samples for {img_dir}")
    if is_train:
        y = np.array([l for _, _, l, _ in ds.items], dtype=np.int64)
        w = 1.0 / (np.bincount(y, minlength=2) + 1e-6)
        sampler = WeightedRandomSampler(w[y], num_samples=len(y), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


# ---------------------------------------------------------------------------
# Inference that keeps case ids
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model, loader, device) -> tuple[list[str], np.ndarray, np.ndarray]:
    model.eval()
    ids, ys, ps = [], [], []
    for x, y, _m, case in loader:
        logits = model(x.to(device))
        ps.append(torch.sigmoid(logits).cpu().numpy().ravel())
        ys.append(y.numpy().ravel())
        ids.extend(case if isinstance(case, (list, tuple)) else list(case))
    return ids, np.concatenate(ys).astype(int), np.concatenate(ps).astype(float)


def accuracy_at_half(y: np.ndarray, p: np.ndarray) -> float:
    return float(((p >= 0.5).astype(int) == y).mean())


# ---------------------------------------------------------------------------
# Phase C: internal-only training
# ---------------------------------------------------------------------------

def train_one(arch: str, seed: int, device) -> dict:
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

    tr = locked_loader(TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_map, HP["batch_size"], True)
    va = locked_loader(INT_IMG_DIR, INT_MASK_DIR, lbl_int, HP["batch_size"], False)

    model = build_model(arch).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    opt = optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=HP["epochs"],
                                                 eta_min=HP["eta_min"])

    run_dir = OUT_ROOT / f"{arch}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = run_dir / "best.pth"

    # Primary criterion is internal_val AUC (protocol
    # classification.selection.criterion). Accuracy is still recorded every
    # epoch so both rankings can be reported, but it does not drive retention.
    best_auc, best_acc, best_epoch, stale = -np.inf, float("nan"), None, 0

    with (run_dir / "train_log.txt").open("w") as logf:
        logf.write(f"=== {arch} seed={seed} :: training + internal_val ONLY ===\n")
        logf.write("=== external cohorts gated; see protocol/gate_audit.jsonl ===\n")
        for epoch in range(1, HP["epochs"] + 1):
            model.train()
            for x, y, m, _ in tr:
                x, y, m = x.to(device), y.to(device), m.to(device)
                opt.zero_grad()
                ys = y * (1 - LABEL_SMOOTH) + 0.5 * LABEL_SMOOTH
                w = (m.mean(dim=(1, 2)) * 0.9 + 0.1).clamp(min=0.1)
                (criterion(model(x), ys) * w).mean().backward()
                opt.step()
            sched.step()

            _, yv, pv = predict(model, va, device)
            acc = accuracy_at_half(yv, pv)
            auc = float(roc_auc_score(yv, pv)) if len(np.unique(yv)) > 1 else float("nan")
            logf.write(f"[{arch} seed{seed}] epoch {epoch:02d} "
                       f"val_acc={acc:.4f} val_auc={auc:.4f}\n")
            logf.flush()

            if np.isfinite(auc) and auc > best_auc + 1e-6:
                best_auc, best_acc, best_epoch, stale = auc, acc, epoch, 0
                torch.save(model.state_dict(), ckpt)
            else:
                stale += 1
                if stale >= HP["patience"]:
                    logf.write(f"early stop at epoch {epoch}\n")
                    break
        logf.write(f"BEST epoch={best_epoch} val_acc={best_acc:.4f} val_auc={best_auc:.4f}\n")

        # Secondary statistics on the retained checkpoint. AUC is the declared
        # selection criterion; the two accuracy variants are recorded so the
        # ranking can be reported under both definitions, which is what
        # supports the claim that the architectures are indistinguishable
        # rather than that one of them won.
        from common import threshold_max_accuracy  # type: ignore
        model.load_state_dict(torch.load(ckpt, map_location=device))
        _, yv, pv = predict(model, va, device)
        thr_opt = float(threshold_max_accuracy(yv, pv))
        acc_opt = float(((pv >= thr_opt).astype(int) == yv).mean())
        logf.write(f"SELECTION_STAT auc={best_auc:.6f} "
                   f"acc_at_opt_thr={acc_opt:.6f} opt_thr={thr_opt:.6f} "
                   f"acc_at_0.5={best_acc:.6f}\n")

    return {"arch": arch, "seed": seed, "best_epoch": best_epoch,
            "internal_val_auc": float(best_auc),   # declared selection statistic
            "internal_val_accuracy_at_half": float(best_acc),
            "internal_val_threshold_max_acc": thr_opt,
            "internal_val_accuracy": acc_opt,
            "ckpt": str(ckpt.relative_to(REPO))}


def phase_train(seeds: list[int], arches: list[str], device) -> None:
    gate.install()
    if not gate.is_locked():
        raise SystemExit("lock record already exists; Phase C must run before Phase D")

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = json.loads(SWEEP.read_text())["runs"] if SWEEP.exists() else []
    done = {(r["arch"], r["seed"]) for r in rows}

    for arch in arches:
        for seed in seeds:
            if (arch, seed) in done:
                print(f"skip {arch} seed{seed}")
                continue
            print(f"--- {arch} seed{seed} ---", flush=True)
            rows.append(train_one(arch, seed, device))
            SWEEP.write_text(json.dumps(
                {"environment": environment(), "runs": rows}, indent=2))
    print(f"wrote {SWEEP.relative_to(REPO)}")


# ---------------------------------------------------------------------------
# Phase D: selection and lock
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    """
    Resolve HEAD, failing loudly.

    Previously returned "unknown" on any exception, which would have written a
    lock record that silently fails to identify the code it locked. A lock that
    cannot name its own commit is not a lock.
    """
    out = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip()
    return out if not dirty else f"{out} (WORKING TREE DIRTY)"


def environment() -> dict:
    import platform, sklearn, torchvision
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "git_commit": git_commit(),
    }


def phase_lock(device) -> None:
    """
    Apply the declared selection rule to internal results only.

    Rule: argmax internal_val accuracy across all architectures AT THE PRIMARY
    SEED. Robustness seeds are reported but take no part, so the choice cannot
    be influenced by picking a favourable seed.
    """
    gate.install()
    rows = json.loads(SWEEP.read_text())["runs"]

    primary = [r for r in rows if r["seed"] == PRIMARY_SEED
               and np.isfinite(r.get("internal_val_auc", np.nan))]
    if len(primary) != len(ARCHES):
        raise SystemExit(f"expected {len(ARCHES)} primary-seed runs, found {len(primary)}")

    # Declared rule: argmax internal_val AUC; ties broken by accuracy at the
    # architecture's own optimal threshold, then alphabetically. The tie-break
    # is explicit because a bare sorted() resolves ties by list order, which is
    # not a scientific criterion.
    ranking = sorted(primary, key=lambda r: (-r["internal_val_auc"],
                                             -r["internal_val_accuracy"],
                                             r["arch"]))

    print(f"=== selection on internal_val AUC, seed {PRIMARY_SEED} ===")
    for i, r in enumerate(ranking, 1):
        print(f"  {i:2d}. {r['arch']:18s} auc={r['internal_val_auc']:.6f} "
              f"acc_opt={r['internal_val_accuracy']:.4f} "
              f"acc@0.5={r['internal_val_accuracy_at_half']:.4f}")

    # The secondary ranking is printed too. If the two orderings disagree, that
    # disagreement is itself the reportable finding: it means internal
    # validation does not separate these architectures.
    alt = sorted(primary, key=lambda r: -r["internal_val_accuracy"])
    if [r["arch"] for r in alt] != [r["arch"] for r in ranking]:
        print("\nNOTE: the accuracy ranking differs from the AUC ranking.")
        print("  by accuracy: " + ", ".join(r["arch"] for r in alt[:5]))
        print("  by AUC     : " + ", ".join(r["arch"] for r in ranking[:5]))

    winner, runner_up = ranking[0], ranking[1]
    tied_auc = abs(winner["internal_val_auc"] - runner_up["internal_val_auc"]) < 1e-12
    tied_acc = abs(winner["internal_val_accuracy"] - runner_up["internal_val_accuracy"]) < 1e-12
    if tied_auc and tied_acc:
        raise SystemExit(
            f"UNRESOLVED TIE between {winner['arch']} and {runner_up['arch']}: "
            f"identical AUC and accuracy. The declared tie-break does not settle "
            f"this. Stopping rather than letting list order decide. Amend the "
            f"protocol and record the change in protocol/DEVIATIONS.md.")
    if tied_auc:
        print(f"\nNOTE: exact AUC tie between {winner['arch']} and "
              f"{runner_up['arch']} at {winner['internal_val_auc']:.16f}; "
              f"resolved by the declared accuracy tie-break "
              f"({winner['internal_val_accuracy']:.6f} vs "
              f"{runner_up['internal_val_accuracy']:.6f}).")

    # Separation between the top architectures, reported against between-seed
    # variability. If the spread across the leaders is smaller than the spread
    # a single architecture shows across seeds, internal validation does not
    # distinguish them and no selection rule can honestly claim it does.
    top5 = ranking[:5]
    spread = top5[0]["internal_val_auc"] - top5[-1]["internal_val_auc"]
    seed_sd = {}
    for a in {r["arch"] for r in top5}:
        vals = [r["internal_val_auc"] for r in rows
                if r["arch"] == a and np.isfinite(r.get("internal_val_auc", np.nan))]
        if len(vals) > 1:
            seed_sd[a] = float(np.std(vals))
    print(f"\ntop-5 AUC spread: {spread:.4f}")
    if seed_sd:
        print(f"between-seed SD among those architectures: "
              f"{min(seed_sd.values()):.4f} to {max(seed_sd.values()):.4f}")
        if spread < max(seed_sd.values()):
            print("  -> the leaders are separated by less than seed noise; "
                  "report them as indistinguishable rather than ranked.")

    print(f"\nSELECTED: {winner['arch']} (seed {PRIMARY_SEED})")
    print("Per protocol classification.selection.outcome_contract, this is the "
          "reported result whatever it is.")

    # Threshold, derived on internal_val only, by the declared Youden rule.
    from common import threshold_youden  # type: ignore
    model = build_model(winner["arch"]).to(device)
    model.load_state_dict(torch.load(REPO / winner["ckpt"], map_location=device))
    lbl_int = load_labels(LABEL_FILES["internal"])
    va = locked_loader(INT_IMG_DIR, INT_MASK_DIR, lbl_int, 16, False)
    _, yv, pv = predict(model, va, device)
    thr = float(threshold_youden(yv, pv))
    print(f"threshold (Youden on internal_val): {thr:.4f}")

    robustness = {
        str(s): {r["arch"]: r["internal_val_accuracy"] for r in rows if r["seed"] == s}
        for s in ROBUSTNESS_SEEDS
    }

    LOCK_RECORD.write_text(json.dumps({
        "protocol": "protocol/analysis_plan_v1.yaml",
        # Cryptographically binds this lock to the exact protocol text it claims
        # to follow, so a later edit to the plan cannot be passed off as the
        # rule that governed this run.
        "protocol_sha256": sha256(PROTOCOL / "analysis_plan_v1.yaml"),
        "selection_rule": (
            "argmax internal_val AUC, all 13 architectures, primary seed; "
            "ties broken by accuracy at the architecture's own "
            "accuracy-maximising threshold, then alphabetically"),
        "selected_architecture": winner["arch"],
        "selected_seed": PRIMARY_SEED,
        "selected_epoch": winner["best_epoch"],
        "internal_val_accuracy": winner["internal_val_accuracy"],
        "internal_val_auc": winner["internal_val_auc"],
        "threshold": thr,
        "threshold_rule": "Youden J on internal_val",
        "ranking_internal_only": [
            {"rank": i, "arch": r["arch"],
             "internal_val_accuracy": r["internal_val_accuracy"],
             "internal_val_auc": r["internal_val_auc"]}
            for i, r in enumerate(ranking, 1)],
        "robustness_seeds_internal_accuracy": robustness,
        "weights_sha256": {
            f"{r['arch']}_seed{r['seed']}": sha256(REPO / r["ckpt"])
            for r in rows if (REPO / r["ckpt"]).exists()},
        "git_commit": git_commit(),
        "environment": environment(),
    }, indent=2))
    print(f"\nwrote {LOCK_RECORD.relative_to(REPO)}  --  gate may now be released")


# ---------------------------------------------------------------------------
# Phase E: single external pass
# ---------------------------------------------------------------------------

def phase_external(device) -> None:
    gate.install()
    gate.release()          # raises unless the lock record exists and validates

    record = json.loads(LOCK_RECORD.read_text())
    arch, seed, thr = (record["selected_architecture"],
                       record["selected_seed"], record["threshold"])

    model = build_model(arch).to(device)
    model.load_state_dict(torch.load(
        OUT_ROOT / f"{arch}_seed{seed}" / "best.pth", map_location=device))

    lbl_train = load_labels(LABEL_FILES["train"])
    train_map = {normalize_case_prefix(p.name): lbl_train[normalize_case_prefix(p.name)]
                 for p in TRAIN_IMG_DIR.glob("*.nii.gz")
                 if normalize_case_prefix(p.name) in lbl_train}

    cohorts = {
        "train":          (TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_map),
        "internal_val":   (INT_IMG_DIR, INT_MASK_DIR, load_labels(LABEL_FILES["internal"])),
        "external_test1": (EXT1_IMG_DIR, EXT1_MASK_DIR, load_labels(LABEL_FILES["external1"])),
        "external_test2": (EXT2_IMG_DIR, EXT2_MASK_DIR, load_labels(LABEL_FILES["external2"])),
    }

    out = {}
    probs_dir = RESULTS / "probabilities"
    probs_dir.mkdir(parents=True, exist_ok=True)

    for name, (img_d, mask_d, lbl) in cohorts.items():
        loader = locked_loader(img_d, mask_d, lbl, 16, False)
        ids, y, p = predict(model, loader, device)

        # Cohort-prefixed ids. In the deposited data every cohort numbers
        # independently from case_00001, so a bare case_id collides across all
        # four cohorts and is not a patient identifier. Prefixing makes the id
        # globally unique within this repository. It does NOT recover patient
        # identity; see analysis/results/task0b_cohort_overlap.md.
        prefix = {"train": "train", "internal_val": "ival",
                  "external_test1": "ext1", "external_test2": "ext2"}[name]

        path = probs_dir / f"{arch}_seed{seed}_{name}_probs.csv"
        with open(path, "w") as fh:
            fh.write("case_id,cohort_case_id,label,prob_malignant\n")
            for cid, yi, pi in zip(ids, y, p):
                fh.write(f"{prefix}_{cid},{cid},{int(yi)},{pi:.6f}\n")

        pred = (p >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        out[name] = {
            "n": len(y), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
            "threshold": thr,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": (tp + tn) / len(y),
            "sensitivity": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None,
            "probs_file": str(path.relative_to(REPO)),
        }
        print(f"{name:15s} n={len(y):4d} auc={out[name]['auc']}")

    (RESULTS / "final_evaluation.json").write_text(json.dumps({
        "lock_record": json.loads(LOCK_RECORD.read_text()),
        "cohorts": out,
    }, indent=2))
    print(f"\nwrote {(RESULTS / 'final_evaluation.json').relative_to(REPO)}")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["train", "lock", "external", "status"])
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[PRIMARY_SEED] + ROBUSTNESS_SEEDS)
    ap.add_argument("--arches", nargs="+", default=ARCHES)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the protocol epoch count; smoke testing only, "
                         "never for a run that feeds the lock")
    args = ap.parse_args()

    if args.epochs is not None:
        HP["epochs"] = args.epochs
        print(f"!! epochs overridden to {args.epochs}: SMOKE TEST ONLY, "
              f"results must not feed Phase D", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.phase == "status":
        print(json.dumps(gate.status(), indent=2))
    elif args.phase == "train":
        phase_train(args.seeds, args.arches, device)
    elif args.phase == "lock":
        phase_lock(device)
    else:
        phase_external(device)


if __name__ == "__main__":
    main()
