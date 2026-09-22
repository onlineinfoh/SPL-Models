#!/usr/bin/env python3
"""Inventory current bytes and historical artifact mappings; no model execution.

This is an inventory made after the experiments, not a contemporaneous lock.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = REPO / "docs/history/2026-09-22-before-scope-correction"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def producer(name):
    if name.startswith("protocol/") or "Dataset001_lungval" in name:
        return "Retained revision experiment; see protocol/README.md and its archived result map"
    if "Dataset000_lung" in name:
        return "nnU-Net Dataset000_lung/2d/fold_all; archived plans.json and fold_all/debug.json"
    if name.startswith("binary_classification/predictions_tight/"):
        return "binary_classification/infer_probs_tight.py at 6475d07; 224 px, seed 67; dataset path correction documented"
    if name.startswith("binary_classification/runs/") or "single_seed_67" in name:
        return "binary_classification/train.py original; 300 px, seed 67, internal-accuracy checkpoint rule"
    if "multiseed_sweep" in name or "runs_locked" in name or "locked_" in Path(name).name:
        return "analysis/code/train_locked.py; earlier mean-internal-accuracy sweep, seeds 67/1234/2025"
    if "nnunet_benchmark_results" in name:
        stem = Path(name).stem
        script = {"metrics_nnunet": "nnunet", "metrics_deeplab": "deeplab", "metrics": "unet"}.get(stem)
        if script is None:
            return "Per-model benchmark documentation; see docs/REPRODUCE.md"
        return f"seg-model-training/benchmarking/{script}_benchmark.py" if script != "nnunet" else "seg-model-training/benchmarking/nnunet_benchmarking.py"
    for prefix, script in (
        ("task0b", "task0b_cohort_overlap.py"), ("task0", "task0_data_check.py"),
        ("task1", "task1_confidence_intervals.py"), ("task2", "task2_threshold_policy.py"),
        ("task3", "task3_calibration.py"), ("task4", "task4_dca.py"),
        ("task5", "task5_model_selection.py"), ("task6", "task6_gt_vs_model_mask.py"),
        ("table2", "make_table2.py"), ("seg_metrics", "seg_metrics_engine.py"),
        ("checkpoint_selection", "summarize_training_logs.py"),
        ("subgroup", "subgroup_and_precision.py"), ("external_cohort_precision", "subgroup_and_precision.py"),
    ):
        if Path(name).name.startswith(prefix):
            return f"analysis/code/{script}; exact inputs and historical configuration in docs/REPRODUCE.md"
    return "Historical retained artifact; generation provenance not fully established"


def main():
    preserved = json.loads((ARCHIVE / "preserved_artifacts.json").read_text())
    artifacts = []
    for name, before in sorted(preserved.items()):
        path = REPO / name
        after = sha256(path)
        if after != before:
            raise RuntimeError(f"historical artifact changed: {name}")
        artifacts.append(dict(path=name, sha256=after, producer=producer(name), unchanged=True))

    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=REPO).decode().split("\0")
    source_paths = {REPO / name for name in tracked if name and Path(name).suffix in (".py", ".sh", ".yaml", ".md")}
    source_paths.update((REPO / "analysis/code").glob("*.py"))
    source_paths.update((REPO / "docs").glob("*.md"))
    source_paths.add(REPO / "protocol/README.md")
    source_paths.add(REPO / ".gitignore")
    configs = [REPO / "analysis/environment_lock.txt"]
    nn = REPO / "seg-model-training/nnunet/nnUNet_results/Dataset000_lung/nnUNetTrainer__nnUNetPlans__2d"
    configs += [nn / "plans.json", nn / "dataset.json", nn / "fold_all/debug.json"]
    weights = sorted((REPO / "binary_classification/runs").glob("*/best.pth"))
    weights += [nn / "fold_all/checkpoint_best.pth",
                REPO / "seg-model-training/Pytorch-UNet/checkpoints/checkpoint_best.pth",
                REPO / "seg-model-training/DeepLabV3Plus-Pytorch/checkpoints/best_deeplabv3plus_mobilenet_lung_os16.pth"]
    inventory = lambda paths: {str(p.relative_to(REPO)): sha256(p) for p in sorted(paths) if p.is_file()}
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Retrospective current-file inventory; not evidence of pre-result locking",
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "baseline_reference": "6475d07",
        "active_segmentation": {"dataset": "Dataset000_lung", "configuration": "2d", "fold": "all"},
        "reported_classification": {"architecture": "densenet121", "seed": 67, "mask_variant": "model",
                                    "historical_inference_px": 224, "threshold": 0.5},
        "corrected_inference": {"pixels": 300, "executed": False, "outputs": "binary_classification/predictions_tight_300"},
        "current_source_sha256": inventory(source_paths), "configuration_sha256": inventory(configs),
        "available_checkpoint_sha256": inventory(weights), "artifacts": artifacts,
        "limitations": [
            "Manuscript/Supplement/Figure 2 unavailable; authors must confirm cell-to-artifact mapping.",
            "Current DeepLab checkpoint is from the later run; original weight provenance unresolved.",
            "U-Net mask-generation configuration at deposit is not fully captured; see result map.",
            "Current source hashes do not prove historical generation by those exact source bytes.",
            "Original 224 px outputs have not been regenerated at 300 px.",
        ],
    }
    out = REPO / "docs/provenance_manifest.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Verified {len(artifacts)} unchanged artifacts; hashed {len(weights)} available checkpoints.")
    print(out)


if __name__ == "__main__":
    main()
