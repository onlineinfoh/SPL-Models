# analysis

Evaluation and statistical analysis for the two-stage segmentation and classification pipeline.
Scripts here read the locked model outputs and produce the reported tables and figures.

## Layout

| Path | Contents |
|---|---|
| `code/` | Analysis scripts |
| `results/` | Generated tables (CSV, JSON, markdown) |
| `figures/` | Calibration and decision-curve figures (PDF and PNG) |
| `logs/training_logs/` | Training logs from both runs |
| `runs_locked/` | Per-run logs from the multi-seed sweep |
| `environment_lock.txt` | Pinned package versions |

## Scripts

Segmentation:

| Script | Output |
|---|---|
| `run_seg_inference.py` | Predicted masks for U-Net and DeepLabv3+ at native resolution |
| `seg_metrics_engine.py` | Per-case Dice, IoU, precision, recall, FPR, HD95, ASSD with bootstrap CIs |
| `make_table2.py` | Table 2 and the paired model comparisons |

Classification:

| Script | Output |
|---|---|
| `task0_data_check.py` | Cohort sizes, class balance, duplicate and range checks |
| `task1_confidence_intervals.py` | Metrics with Wilson and bootstrap intervals on class-specific denominators |
| `task2_threshold_policy.py` | Operating point under fixed, accuracy-maximising and Youden criteria |
| `task3_calibration.py` | Calibration intercept, slope, Brier score, binned curves |
| `task4_dca.py` | Decision-curve analysis |
| `task5_model_selection.py` | Architecture ranking and DeLong comparisons |
| `task6_gt_vs_model_mask.py` | Manual versus automatic mask pipeline |
| `subgroup_and_precision.py` | Subgroup performance by lesion size, achieved precision per cohort |

Training and utilities:

| Script | Purpose |
|---|---|
| `train_locked.py` | Multi-seed training sweep over all candidate architectures |
| `cached_dataset.py` | RAM-cached dataset used to make the sweep tractable |
| `verify_cached_dataset.py` | Checks the cached dataset against the original implementation |
| `summarize_training_logs.py` | Parses training logs and reports the epoch each selection rule would pick |
| `common.py` | Shared paths, Wilson and bootstrap intervals, DeLong test |

## Running

```bash
PY=~/venvs/prism/bin/python

$PY analysis/code/run_seg_inference.py
$PY analysis/code/seg_metrics_engine.py
$PY analysis/code/make_table2.py

bash analysis/code/run_all.sh

$PY analysis/code/verify_cached_dataset.py
$PY analysis/code/train_locked.py --seeds 67 1234 2025
$PY analysis/code/summarize_training_logs.py
```

## Conventions

Segmentation metrics are computed on the lesion class only, at native image resolution.
FPR is `FP / (FP + TN)`, so the denominator is every true background pixel.
HD95 and ASSD are in pixels on the native grid and are undefined when either mask is empty.

Classification intervals use the class-specific denominator: sensitivity over malignant cases, specificity over benign cases.

All bootstrap procedures use 2000 resamples with seed 20260904.
Training seeds are 67, 1234 and 2025.
Since one image is analysed per patient, image-level and patient-level resampling are equivalent.
