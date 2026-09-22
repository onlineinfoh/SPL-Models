# Result-to-code map

Every result reported in the manuscript, mapped to the exact script, configuration, seed, mask variant and inference resolution that produced it.

This repository is the implementation that generated the **originally submitted** results.
Where the code and the reported numbers disagree, or where a source file is not in the repository, that is stated in [Known gaps](#known-gaps) rather than omitted.

Reported results come from a single run: **seed 67**, 13 architectures, classifier training at 300 px, classifier inference at **224 px**.
See [Known gaps](#known-gaps) item 2 for the resolution discrepancy.

## Pipeline

```
data/<cohort>/img_*            manual lesion masks (seg_*/labelsTr)
      |
      |  Stage 1: nnU-Net v2 segmentation
      v
data/<cohort>/img_*_model      automatic masks, used as the "model" variant
      |
      |  Stage 2: lesion-ROI crop + 10% halo, 2-channel (image, mask)
      v
binary_classification/predictions_tight/<arch>/<split>[_model]_seed67_probs.txt
      |
      v
analysis/results/*.csv  ->  manuscript tables
```

## Stage 1: segmentation

| Item | Value |
|---|---|
| Framework | nnU-Net v2 |
| Dataset | `Dataset000_lung` |
| Configuration | `2d` |
| Fold | `all` |
| Checkpoint | `checkpoint_best.pth` |
| Batch size | 2 |
| Patch size | 896 x 1792 |
| Spacing | 1.0 x 1.0 |
| Network | PlainConvUNet (nnU-Net default) |
| Training command | `nnUNetv2_train Dataset000_lung 2d nnUNetTrainer__nnUNetPlans__2d -f all` |
| Plans | `seg-model-training/nnunet/nnUNet_results/Dataset000_lung/nnUNetTrainer__nnUNetPlans__2d/plans.json` |
| Training log | same directory, `fold_all/` |
| Notes | `seg-model-training/nnunet/README_SPL.md` |

**Fold `all` has no held-out validation.**
In nnU-Net v2, fold `all` sets `val_keys = tr_keys` (`nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:565-569`), so `fold_all/validation/` contains the same 600 cases used for fitting.
The Dice value printed during training is therefore **training-set Dice**, not internal-validation Dice.
Evaluation on the 257-case Center 1 tuning cohort and on the two external cohorts is computed separately, by the scripts below, and those values are held-out.

Baseline segmentation models (U-Net, DeepLabv3+) are under `seg-model-training/benchmarking/`.

## Stage 2: classification

### Training

| Item | Value | Location |
|---|---|---|
| Script | `binary_classification/train.py` | |
| Seed | 67 | `train.py:566` (`--seeds` default `[67]`) |
| Architectures | 13 | `infer_probs_tight.py:27-33` |
| Training resolution | **300 x 300** | `train.py:46` (`IMG_SIZE = 300`) |
| Input | 2 channels: image, mask | |
| ROI crop | lesion bounding box + 10% halo | `HALO_FRAC = 0.10` |
| Checkpoint rule | highest **internal-validation accuracy** | `train.py:690` |
| Checkpoints | `binary_classification/runs/<arch>/best.pth` | not deposited, see gap 4 |
| Logs | `binary_classification/runs/<arch>/train_log.txt`, `analysis/logs/training_logs/` | |

The checkpoint rule is accuracy on the internal-validation cohort, not AUC.
`train.py` also evaluates both external cohorts inside the per-epoch loop and writes their AUCs to the training log (`train.py:652-653`).
Those values are monitoring output; they are not read by the checkpoint rule at `train.py:690`, which compares `val_acc` only.

### Inference

| Item | Value | Location |
|---|---|---|
| Script | `binary_classification/infer_probs_tight.py` | |
| Inference resolution | **224 x 224** | `infer_probs_tight.py:37` (`IMG_SIZE = 224`) |
| Mask variants | `gt` (manual), `model` (automatic) | `iter_mask_variants`, `infer_probs_tight.py:162-170` |
| Operating threshold | accuracy-maximising on internal validation | `_best_threshold_from_rows`, `infer_probs_tight.py:214` |
| DenseNet121 thresholds | 0.5483 manual, 0.5000 automatic | |
| Outputs | `binary_classification/predictions_tight/<arch>/<split>[_model]_seed67_probs.txt` | |
| Driver | `binary_classification/run_tight_pipeline.sh` | |

The threshold is derived on the internal-validation cohort and applied unchanged to both external cohorts.
The `model` variant is selected automatically when a sibling `<img_dir>_model` directory exists.

## Reported result to script

Run from `analysis/code/`. Outputs land in `analysis/results/`.

| Reported result | Script | Output file |
|---|---|---|
| Table 2, segmentation Dice by cohort | `make_table2.py` | `table2_corrected.csv`, `table2_corrected.md` |
| Table 2 paired tests | `make_table2.py` | `table2_paired_tests.csv` |
| Per-case segmentation metrics | `seg_metrics_engine.py` | `seg_metrics_per_case.csv`, `seg_metrics_summary.json` |
| Boundary metrics (HD95, ASSD) | `seg_metrics_engine.py` | `table2b_boundary_metrics.csv`, `.md` |
| Table 3, DenseNet121 metrics with CIs | `task1_confidence_intervals.py` | `task1b_densenet121_corrected_metrics_with_ci.csv` |
| Wilson CI arithmetic check | `task1_confidence_intervals.py` | `task1a_wilson_arithmetic_check.csv`, `task1a_ci_denominator_check.csv` |
| Operating thresholds | `task2_threshold_policy.py` | `task2_thresholds_derived_on_internal_val.csv` |
| Threshold policy comparison | `task2_threshold_policy.py` | `task2_threshold_policy_comparison.csv`, `task2_maxacc_vs_youden_all_architectures.csv`, `task2_youden_vs_used_deltas.csv` |
| Calibration | `task3_calibration.py` | `task3_calibration_summary.csv`, `task3_calibration_curves_10bin.csv` |
| Decision curve analysis | `task4_dca.py` | `task4_dca_curves.csv`, `task4_dca_net_benefit_at_pt050.csv` |
| 13-architecture comparison | `task5_model_selection.py` | `task5_architecture_ranking.csv` |
| DeLong tests between architectures | `task5_model_selection.py` | `task5_delong_densenet121_vs_efficientnetb1.csv`, `task5_delong_external_test1_all_vs_densenet121.csv` |
| Manual vs automatic mask comparison | `task6_gt_vs_model_mask.py` | `task6_gt_vs_automatic_mask.csv` |
| Subgroup analysis by lesion size | `subgroup_and_precision.py` | `subgroup_by_lesion_size.csv`, `subgroup_and_precision.json` |
| External precision | `subgroup_and_precision.py` | `external_cohort_precision.csv` |
| Data integrity checks | `task0_data_check.py` | `task0_data_integrity_check.csv`, `task0_problems.txt` |
| Training-log summary | `summarize_training_logs.py` | `checkpoint_selection_summary.csv`, `.json` |

Shared helpers: `common.py` (paths, cohorts, mask variants, metrics, DeLong), `cached_dataset.py`, `run_all.sh`.

All three segmentation models are scored by one engine, `seg_metrics_engine.py`, on the lesion class only, at native resolution.
The derivation of the corrected Table 2 false-positive rates, including the macro-averaging defect in the original DeepLabv3+ evaluation, is in [`table2_fpr_artifact_proof.md`](table2_fpr_artifact_proof.md).

### Architecture selection

DenseNet121 was selected on **internal-validation accuracy**, the rule implemented at `train.py:690`.
On that criterion it ranks **first of the 13 architectures**.
The supporting table is `analysis/results/task5_architecture_ranking.csv`, column `internal_val_accuracy`, produced by `task5_model_selection.py`.

For the automatic-mask variant at seed 67, that file also records the external results: highest accuracy and AUC on External Test 2, and third of 13 on External Test 1 on both metrics.
The selection rule reads internal-validation accuracy only; the external columns are reported outcomes, not selection inputs.

## Known gaps

These are stated so that a reader attempting reproduction is not misled.

1. **`infer_probs_tight.py` data root.**
   As originally deposited, `DATA_CROP` pointed at `new_data/`, a directory that is not present in the repository and never was.
   The cohort images are under `data/`, whose layout matches the script's `SPLITS` table exactly (600 / 257 / 108 / 94 cases).
   The path has been corrected to `data/` so the script runs against the deposited data.
   This is the only executable line changed in this revision; it is logged in `docs/CHANGES_AND_REMOVALS.md`.

2. **Training and inference resolution differ.**
   Training used 300 x 300 (`train.py:46`); inference used 224 x 224 (`infer_probs_tight.py:37`).
   This was not intended and was not noticed before submission.
   All reported classification numbers were produced with 224 x 224 inference, and the manuscript states this.
   No 300 x 300 inference results are reported, and none are deposited.

3. **nnU-Net training-time Dice is training-set Dice.**
   See the Stage 1 note above on fold `all`.
   The reported tuning-cohort and external Dice values are computed separately by `seg_metrics_engine.py` and are held-out.

4. **Model weights are not in the repository.**
   `.gitignore` excludes `*.pth`. Weights are available on reasonable request.
   Without them, the `analysis/code/task*.py` scripts still run, because they read the deposited probability files in `binary_classification/predictions_tight/`.
   Re-running `infer_probs_tight.py` requires the checkpoints.

5. **Patient images, masks and labels are restricted.**
   `data/*` and `binary_classification/labels/*` are excluded.
   See `data/README.md`.

6. **Figure 2 has no source file in this repository.**
   It was produced outside the repository and the generating file is not available.

## Environment

Python 3.11.14, torch 2.13.0+cu126, torchvision 0.28.0+cu126, CUDA 12.6, nnunetv2 2.8.1, numpy 1.26.4, scikit-learn 1.9.0, pandas 1.5.3, nibabel 5.4.2, SimpleITK 2.5.6, opencv-python 4.11.0.86.
Hardware: single NVIDIA GeForce RTX 4090.

`torch.backends.cudnn.deterministic` was left at `False`, so cuDNN kernel selection is not pinned.
Runs are reproducible in distribution rather than bitwise.

Bootstrap procedures use 2000 resamples with fixed seed 20260904.
