# Result-to-code and configuration map

This map covers the deposited repository outputs. The manuscript, Supplement
and Figure 2 source were not provided, so correspondence to every manuscript
cell remains for the authors to confirm using MANUSCRIPT_HANDOFF.md. A mapped
file is not evidence that its generation environment or selection timeline was
locked contemporaneously.

## Versions and status

| Version | Meaning |
|---|---|
| `6475d07` / preceding deposit `a1e59eb` | Original scripts and stored seed-67 results; nnU-Net Dataset000/fold_all; classifier 300 px training and 224 px inference |
| ~~`9f18f16` revision pipeline as the reported model~~ | Retained alternative experiment: Dataset001/fold_0, corrected classifier, EfficientNet-B0 selected by the internal-AUC rule |
| Current working tree | Original nnU-Net/checkpoints, future training text output suppressed, classification inference/Grad-CAM corrected to 300 px; results not regenerated |

The archive [manifest](history/2026-09-22-before-scope-correction/manifest.json)
identifies the pre-edit copies. `provenance_manifest.json` records the exact
current source, configuration, checkpoint and artifact hashes. A current hash
identifies the current bytes; it does not prove which bytes generated a
historical output when no run manifest was recorded. Such gaps are stated below.
No commit was created or pushed by this update.

## Segmentation: original nnU-Net

- Model: `seg-model-training/nnunet/nnUNet_results/Dataset000_lung/nnUNetTrainer__nnUNetPlans__2d/fold_all/checkpoint_best.pth`.
- Configuration: sibling `plans.json`, `dataset.json`, `dataset_fingerprint.json`; `fold_all/debug.json` and the December 2025 training log.
- 2d PlainConvUNet, patch 896×1792, batch 2, 1000 epochs, SGD, initial LR 0.01, weight decay 3e-5, foreground oversampling 0.33, CTNormalization, spacing 1×1.
- Original debug record: torch 2.9.1+cu128. This differs from the deposited later analysis environment.
- `seg-model-training/benchmarking/nnunet_benchmarking.py::main` loads **use_folds=("all",)** and `checkpoint_best.pth`.
- `seg-model-training/nnunet/predict_all_cohorts.sh` now explicitly selects dataset 0/fold all and writes to `predicted_masks_fold_all/`. No prediction command was executed. Existing masks are not overwritten.

~~Three-fold training / internal-validation Dice during training~~ → **fold all;
training-set pseudo-Dice / training-set evaluation Dice**. The original trainer
assigns `val_keys = tr_keys` for fold all (`nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py`).
The final framework label "Mean Validation Dice" refers to the training cases
for this fold. Preserve the raw logs and correct the interpretation in prose.

The separately evaluated Center 1 tuning row is distinct and must not be renamed
training Dice. The original shared-metric means are:

| Cohort | N | Dice | Source |
|---|---:|---:|---|
| Training | 600 | 0.9824054791 | `analysis/results/seg_metrics_summary.json` |
| Center 1 tuning | 257 | 0.9202934394 | same |
| External Test 1 | 108 | 0.9134765299 | same |
| External Test 2 | 94 | 0.9149197328 | same |

These files are unchanged, not recalculated. The later fold-0 experiment has
different external Dice (0.90838254 / 0.90368419) and is not interchangeable.

## Segmentation result mapping

| Result/artifact | Exact producing code and inputs | Configuration / status |
|---|---|---|
| Original raw nnU-Net benchmark | `seg-model-training/benchmarking/nnunet_benchmarking.py` → `binary_classification/nnunet_benchmark_results/metrics_nnunet.txt` | Dataset000/fold_all; foreground metrics; normal-approximation CIs |
| Original raw U-Net benchmark | `seg-model-training/benchmarking/unet_benchmark.py` → `binary_classification/nnunet_benchmark_results/metrics.txt` | `Pytorch-UNet/checkpoints/checkpoint_best.pth`; native inputs; per-model metric definitions |
| Original raw DeepLab benchmark | `seg-model-training/benchmarking/deeplab_benchmark.py` → `binary_classification/nnunet_benchmark_results/metrics_deeplab.txt` | Mobilenet, output stride 16, 512 px; background/lesion macro metrics; not shared Table 2 |
| Table 2, overlap metrics and CIs | `analysis/code/seg_metrics_engine.py` → `analysis/results/seg_metrics_summary.json`, `seg_metrics_per_case.csv`; `make_table2.py` → `table2_corrected.{csv,md}` | Original saved nnU-Net `data/*/*_model` masks; U-Net/DeepLab `analysis/masks_locked/`; foreground only, native grid, 2000 bootstrap resamples, seed 20260904 |
| Table 2b, HD95/ASSD | Same metric engine and `make_table2.py` → `table2b_boundary_metrics.{csv,md}` | Native-grid pixels; median/IQR; empty-mask distances undefined |
| Segmentation paired tests | `make_table2.py` → `table2_paired_tests.csv` | Paired per-case metrics, Wilcoxon tests |
| ~~Revision segmentation tables~~ | `protocol/results/analysis/table2*`, `seg_metrics*` | Dataset001/fold_0 masks plus later baseline choices; retained, excluded from original result set |

Original U-Net documentation records 350 epochs, batch 1, RMSprop LR 5e-4,
validation split 0. Its training set was reused for training-time validation.
The locally modified DeepLab training code and its current checkpoint include
the later held-out-validation run. The earlier DeepLab weights were overwritten
according to the revision record; the present weights cannot be claimed as the
original weights merely because the filename is unchanged. Original saved masks
and tables remain. Full checkpoint-to-original-table regeneration is unresolved.

A second provenance gap: `run_seg_inference.py` at `6475d07` resized U-Net to
512 px, whereas the present helper uses native resolution. Do not claim that
an exact checkout of that earlier helper reproduces native-resolution U-Net
results. The saved mask/table hashes identify the available results; the
original command/configuration for mask creation was not fully captured.

## Classification configuration and outputs

| Result/artifact | Code and inputs | Configuration / status |
|---|---|---|
| Original per-epoch training / retained checkpoints | `binary_classification/train.py`; `binary_classification/runs/<arch>/best.pth`, `train_log.txt`; archived copies in `analysis/logs/training_logs/single_seed_67/` | 13 architectures, seed 67, 300 px, 40 epochs maximum, batch 16, LR 1e-4, AdamW, patience 8; checkpoint retained by internal **accuracy** at threshold 0.5 |
| Historical per-case probabilities and summary | `binary_classification/infer_probs_tight.py` at `6475d07` (dataset directory subsequently corrected from `new_data` to `data`); `predictions_tight/<arch>/*seed67_probs.txt`, `grand_summary.txt` | 224 px inference; original checkpoints; mask-specific maximum-tuning-accuracy threshold |
| Classification table / Table 3 candidate | `task1_confidence_intervals.py` + `common.py` → `analysis/results/task1b_densenet121_corrected_metrics_with_ci.csv` | Stored 224 px DenseNet121 seed-67 probabilities; manual threshold 0.5483, automatic threshold 0.5000; Wilson class-specific denominators, DeLong and bootstrap AUC CIs |
| CI arithmetic audit | `task1_confidence_intervals.py` → `task1a_*.csv` | Historical examples, not new model runs |
| Threshold sensitivity | `task2_threshold_policy.py` → `task2_*.csv` | Fixed 0.5, maximum accuracy and Youden J; derived on tuning predictions only |
| Calibration tables/plots | `task3_calibration.py` → `task3_*.csv`, `analysis/figures/task3_*` | Historical predictions; joint/single calibration fits as implemented in `common.py` |
| Decision curves | `task4_dca.py` → `task4_*.csv`, `analysis/figures/task4_*` | Historical probabilities; threshold-probability curve; separate net benefit at 0.50 |
| Architecture ranking / DeLong comparisons | `task5_model_selection.py` → `task5_*.csv` | All 13 seed-67 models, both mask variants; retrospective comparisons, not proof of prior selection |
| Manual vs automatic mask comparison | `task6_gt_vs_model_mask.py` → `task6_gt_vs_automatic_mask.csv` | Paired cases, each variant's tuning-derived maximum-accuracy threshold |
| Lesion-size subgroups / precision | `subgroup_and_precision.py` → `subgroup_*.csv`, `subgroup_and_precision.json`, `external_cohort_precision.csv` | Historical automatic-mask probabilities, threshold 0.5000; per-cohort GT lesion-area tertiles |
| Cohort counts / probability integrity | `task0_data_check.py` → `task0_data_integrity_check.csv` | Within-cohort checks; not patient identity verification |
| Cross-cohort image overlap | `task0b_cohort_overlap.py` → `task0b_cohort_overlap.{json,md}` | Image-content checks; cannot establish absence of patient overlap |
| Checkpoint selection audit | `summarize_training_logs.py` → `checkpoint_selection_summary.{csv,json}` | Original logs, retained epochs versus internal/external maxima |
| Earlier multi-seed sweep | `train_locked.py` → `analysis/runs_locked/`, `locked_sweep_results.json`, `locked_selection_and_external.json` | Seeds 67/1234/2025, internal-only training, mean internal accuracy selection; separate experiment |
| ~~Later locked AUC sweep and calibrated outputs~~ | `protocol/run_protocol.py`, `export_locked_predictions.py`, `calibrate_locked_model.py` → `protocol/results/` | Separate 39-run experiment; EfficientNet-B0 seed 67; threshold 0.5054240823; not original DenseNet121 results |
| Historical Grad-CAM | `binary_classification/heatmap.py` at `6475d07` / archived version | Original 224 px path; current code defaults to 300 px |
| Corrected inference / Grad-CAM | Current `infer_probs_tight.py`, `heatmap.py` | 300 px; `predictions_tight_300/`, `heatmaps_tight_300/`; **not run**, no new metrics or thresholds |
| Figure 2 | No source provided | Colleague must align diagram to original fold_all segmentation and distinguish historical 224 px results from pending 300 px outputs |

Bootstrap defaults: 2000 resamples, seed 20260904. The existing task1 and subgroup
scripts retain historical thresholds; simply pointing them at 300 px files will
not establish the correct new operating point. Those values must be reconciled
after new tuning inference before revised classification tables are published.

## Reported seed and model comparison

The authors have chosen to report the verified **original seed-67 automatic-mask
analysis**. Across all 13 architectures:

| Metric | Center 1 tuning | External Test 1 | External Test 2 |
|---|---|---|---|
| Accuracy | 0.856031 (rank 1) | 0.814815 (rank 3) | 0.819149 (rank 1) |
| AUC | 0.905629 (rank 2) | 0.889201 (rank 3) | 0.864566 (rank 1) |

The operating threshold is 0.5000, derived by maximum tuning accuracy for the
automatic-mask variant. The values above use the deposited **224 px inference**
probabilities and seed-67 original checkpoints. They have been verified by
recomputing metrics from those saved probabilities, not by rerunning models.

DenseNet121 had the highest internal accuracy and the highest External Test 2
accuracy and AUC; it was third on External Test 1 in both metrics. Do not shorten
this to “best internally and externally” without specifying the metric and
cohort. Internal AUC was second, not first. The manuscript-ready paragraph is in
[MANUSCRIPT_HANDOFF.md](MANUSCRIPT_HANDOFF.md); the full comparison and source
hashes are in [selection_audit/](selection_audit/README.md).

No different seed is asserted. These observed rankings do not establish a
contemporaneous selection decision or independence of external evaluation.
Correcting inference to 300 px requires new results before the same ranking can
be claimed at the corrected resolution.

## Remaining author confirmations

Confirm manuscript table/figure filenames and row-to-artifact correspondence;
establish the source of original DeepLab
weights if regeneration is needed; and confirm patient identities and clinical
analyses not present in this repository. These gaps are not marked complete.
