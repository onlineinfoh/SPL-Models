# Result-to-code map

Every reported number traced to the code, configuration and data that produced
it. This answers the requirement in Reviewer Comment 3 to "identify a single
version-locked analysis pipeline, map every reported result to the exact
code/configuration that generated it, and explicitly mark any superseded code or
descriptions as such."

## How to read this

Each row gives a reported artifact, the script that writes it, the inputs it
consumes, and its status:

| Status | Meaning |
|---|---|
| **CURRENT** | Produced by the locked re-analysis. This is what the manuscript should report. |
| **SUPERSEDED** | Produced by the original December 2025 run. Retained as the audit record. Not reported. |


Nothing is deleted. A superseded artifact stays in the repository so the two
runs can be compared.

## Environment

Two environments exist and both are declared, because the repository has always
had two and the manuscript declared one.

| Environment | Used for | Status |
|---|---|---|
| `~/venvs/prism` | Everything in the locked re-analysis | Working |
| `nnunet-env` | The original nnU-Net run (torch 2.9.1+cu128) | **Non-functional.** Created at a different path and later moved; console-script shebangs point at a missing interpreter and `import nnunetv2` fails. The published nnU-Net run cannot be reproduced in the environment that produced it. See [DEVIATIONS.md](../protocol/DEVIATIONS.md) D1. |

Pinned versions: [`analysis/environment_lock.txt`](../analysis/environment_lock.txt).

## The locked pipeline

The protocol is [`protocol/analysis_plan_v1.yaml`](../protocol/analysis_plan_v1.yaml).
Its SHA256 is recorded inside `protocol/locked_pipeline.json`, so the lock is
bound to the exact plan text it claims to follow.

```bash
PY=~/venvs/prism/bin/python

# --- Stage 1: segmentation -------------------------------------------------
# Build a dataset with a genuine held-out validation split. fold_all cannot be
# used: it sets val_keys = tr_keys, so the published run validated on its own
# training data.
$PY seg-model-training/nnunet/prepare_dataset_with_validation.py
bash seg-model-training/nnunet/preprocess_with_original_plans.sh
nnUNetv2_train 1 2d 0

# U-Net baseline: internal_val as the validation set, replacing --validation 0
$PY seg-model-training/unet_train_with_validation.py

# DeepLabv3+: --lung_val_img_dir replaces the all_indices train==val default
cd seg-model-training/DeepLabV3Plus-Pytorch && $PY main.py --dataset lung \
  --lung_img_dir ../../data/train/imagesTr --lung_mask_dir ../../data/train/labelsTr \
  --lung_val_img_dir ../../data/val/img_v --lung_val_mask_dir ../../data/val/seg_v \
  --num_classes 2 --crop_size 512 --batch_size 4 --output_stride 16 \
  --total_itrs 30000 --lr 0.01 --val_interval 200

# --- Stage 2: classification, three phases ---------------------------------
$PY protocol/run_protocol.py train      # Phase C: internal only, gate closed
$PY protocol/run_protocol.py lock       # Phase D: apply the rule, write the lock
$PY protocol/run_protocol.py external   # Phase E: single external pass

# --- Evaluation ------------------------------------------------------------
$PY analysis/code/seg_metrics_engine.py
$PY analysis/code/make_table2.py
bash analysis/code/run_all.sh
```

## Stage 1: segmentation

| Reported artifact | Script | Inputs | Output | Status |
|---|---|---|---|---|
| Table 2, segmentation metrics | `analysis/code/make_table2.py` via `seg_metrics_engine.py` | predicted masks, four cohorts | `analysis/results/table2_corrected.{csv,md}` | **SUPERSEDED** — built on checkpoints selected against training data |
| Table 2b, boundary metrics | `analysis/code/make_table2.py` | same | `analysis/results/table2b_boundary_metrics.{csv,md}` | **SUPERSEDED**, same reason |
| Paired model comparisons | `analysis/code/make_table2.py` | per-case metrics | `analysis/results/table2_paired_tests.csv` | **SUPERSEDED** |
| Per-case metrics | `analysis/code/seg_metrics_engine.py` | masks | `analysis/results/seg_metrics_per_case.csv` | **SUPERSEDED** |
| nnU-Net training record | stock nnU-Net | `Dataset001_lungval`, fold 0 | `nnUNet_results/Dataset001_lungval/.../fold_0/` | **CURRENT** — 1000 epochs, held-out Dice 0.9201 |
| U-Net weights | published run, **not retrained** | train 600 | `seg-model-training/Pytorch-UNet/checkpoints/` | **CURRENT** — the checkpoint-selection defect is inert for this arm: lr reached 0 at epoch 45 of 350, so 306 epochs ran frozen with train Dice drift 0.0119 (sd 0.002). Every candidate checkpoint is the same weights. See [DEVIATIONS.md](../protocol/DEVIATIONS.md) D6 |
| U-Net retrain attempts | `seg-model-training/unet_train_with_validation.py` | train 600 / internal_val 257 | `unet_locked/train_log*.txt` | **SUPERSEDED** — three attempts, all collapsed by divergence or learning-rate decay. Logs retained as the audit record; none is reported |
| DeepLabv3+ training record | patched `main.py` | train 600 / internal_val 257 | `seg-model-training/DeepLabV3Plus-Pytorch/checkpoints/` | **CURRENT** — 30000 iterations, Mean IoU 0.742 |
| Predicted masks, nnU-Net | `seg-model-training/nnunet/predict_all_cohorts.sh` | retrained checkpoint | `predicted_masks_v2/` (gitignored, regenerable) | **CURRENT** — 1059 masks |
| Predicted masks, U-Net and DeepLabv3+ | `analysis/code/run_seg_inference.py` | retrained checkpoints | `analysis/masks_locked_v2/` (gitignored, regenerable) | **CURRENT** — 1059 each |
| Table 2 | `seg_metrics_engine.py` + `make_table2.py` | retrained nnU-Net, retrained DeepLabv3+, published U-Net | `protocol/results/analysis/table2_corrected.{csv,md}` | **CURRENT** — nnU-Net 0.917 / 0.908 / 0.904 on tuning, Center 2, Center 3. Mask source is stated per model; set via `SPL_NNUNET_PRED`, `SPL_MASKS_LOCKED`, `SPL_UNET_MASKS` |

All three published segmentation arms selected their checkpoint on training
data. The reviewer identified this for U-Net only:

| Arm | Defect | Location |
|---|---|---|
| nnU-Net | `fold_all` sets `val_keys = tr_keys` | `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:565-569` |
| U-Net | `--validation 0` uses the train set as val | `Pytorch-UNet/train.py:63-68` |
| DeepLabv3+ | train and val both built from `all_indices` | `DeepLabV3Plus-Pytorch/main.py:180-184` |

Metric values in Table 2 were computed per cohort and are not themselves
invalid; what was flawed is the checkpoint selection behind them.

## Stage 2: classification

| Reported artifact | Script | Inputs | Output | Status |
|---|---|---|---|---|
| Architecture sweep, 13 x 3 seeds | `protocol/run_protocol.py train` | train 600, internal_val 257 | `protocol/results/internal_sweep.json` | **CURRENT** |
| Model selection + threshold | `protocol/run_protocol.py lock` | internal sweep only | `protocol/locked_pipeline.json` | **CURRENT** |
| External evaluation | `protocol/run_protocol.py external` | locked model, four cohorts | `protocol/results/final_evaluation.json` | **CURRENT** (manual masks) |
| Per-case probabilities | same | same | `protocol/results/probabilities/*.csv` | **CURRENT** (manual masks) |
| Automatic-mask evaluation | `export_locked_predictions.py --nnunet-masks predicted_masks_v2` | retrained nnU-Net masks | `protocol/results/predictions_locked/` | **CURRENT** |
| Seed robustness | `export_locked_predictions.py --seed {1234,2025}` | locked architecture, post-lock | `protocol/results/predictions_robustness/` | **CURRENT** |
| Probability calibration | `analysis/code/calibrate_locked_model.py` | tuning cohort only | `protocol/results/predictions_calibrated/` | **CURRENT** |
| Calibrated calibration + DCA | `task3_calibration.py`, `task4_dca.py` | calibrated probabilities | `protocol/results/analysis_calibrated/`, `figures_calibrated/` | **CURRENT** |
| Grad-CAM | `binary_classification/heatmap.py` at 300 px, locked threshold | locked checkpoint | `protocol/results/heatmaps/` (gitignored, regenerable) | **CURRENT** — 2118 overlays |
| Hold-out audit trail | `protocol/gate.py` | — | `protocol/gate_audit.jsonl` | **CURRENT** |
| Gate verification | `protocol/test_gate.py` | — | 32/32 checks | **CURRENT** |
| Old sweep, mean-across-seeds rule | `analysis/code/train_locked.py` | internal only | `analysis/results/locked_selection_and_external.json` | **SUPERSEDED** — ranks by mean across seeds, not the declared rule; records `selected_arch: efficientnet_b0` |
| Original 13-architecture run | `binary_classification/train.py` | all four cohorts | `binary_classification/predictions_tight/`, `analysis/logs/training_logs/single_seed_67/` | **SUPERSEDED** — logged external metrics every epoch |

### Reported model

The reported model is **DenseNet121**, as in the original submission.
The internal-only re-analysis below is a sensitivity analysis supporting the
post-selection framing, not a change to the reported model.

| Sensitivity analysis | Value |
|---|---|
| Architecture returned by the internal-only rule | `efficientnet_b0` |
| Seed | 67 (primary; 1234 and 2025 are robustness only) |
| Retained epoch | 4 of 40 |
| Selection rule | argmax internal_val AUC across all 13 architectures |
| Threshold | 0.5054, Youden J on internal_val only |
| Record | `protocol/locked_pipeline.json` |

DenseNet121 ranks 2nd at the primary seed and 7th averaged across seeds. It is
**not** the model the declared internal-only rule selects. See
[DEVIATIONS.md](../protocol/DEVIATIONS.md) D5.

## Statistics and figures

The reported model is DenseNet121, so the statistics under `analysis/results/`
are the reported ones and are CURRENT. The parallel set under
`protocol/results/analysis*/` belongs to the EfficientNet-B0 sensitivity
analysis. The same scripts produce both; the source is selected with
`SPL_PRED_DIR` and `SPL_ARCH`, and with no environment set they produce the
DenseNet121 results.

| Reported artifact | Script | Output (reported, DenseNet121) | Status |
|---|---|---|---|
| Cohort integrity | `task0_data_check.py` | `analysis/results/task0_data_integrity_check.csv` | CURRENT but incomplete — within-split only, see below |
| Cross-cohort overlap | `task0b_cohort_overlap.py` | `analysis/results/task0b_cohort_overlap.{json,md}` | **CURRENT** |
| Metrics with CIs | `task1_confidence_intervals.py` | `analysis/results/task1b_*_metrics_with_ci.csv` | **CURRENT** |
| Threshold policy | `task2_threshold_policy.py` | `analysis/results/task2_*.csv` | **CURRENT** |
| Calibration + Figure | `task3_calibration.py` | `analysis/results/task3_*.csv`, `analysis/figures/task3_*` | **CURRENT** |
| Decision curves + Figure | `task4_dca.py` | `analysis/results/task4_*.csv`, `analysis/figures/task4_*` | **CURRENT** |
| Architecture ranking, DeLong | `task5_model_selection.py` | `analysis/results/task5_*.csv` | **CURRENT** for the reported model. Cannot be re-run under the hold-out protocol, which scores only the selected model externally |
| Manual vs automatic mask | `task6_gt_vs_model_mask.py` | `analysis/results/task6_gt_vs_automatic_mask.csv` | **CURRENT** |
| Subgroup by lesion size | `subgroup_and_precision.py` | `analysis/results/subgroup_by_lesion_size.csv` | **CURRENT** |
| Checkpoint-selection audit | `summarize_training_logs.py` | `analysis/results/checkpoint_selection_summary.csv` | **CURRENT** |
| *Sensitivity analysis, all of the above* | same scripts, `SPL_ARCH=efficientnet_b0` | `protocol/results/analysis*/` | **CURRENT** as a sensitivity analysis, not as reported results |

`task0_data_check.py` reports "all integrity checks passed", but it only tests
`case_id.duplicated()` *within* each split. It never compared cohorts.
`task0b_cohort_overlap.py` runs that comparison and finds no image shared with
either external cohort, and 11 byte-identical duplicates inside the training
cohort (600 images, 589 unique acquisitions).

## Figure 2

Figure 2 is not present in this repository. `analysis/figures/` contains only
the calibration and decision-curve plots. The reviewer asked for Figure 2 to be
corrected where necessary; it cannot be checked against code until the source
that generates it is deposited.

## Disclosures

- [`DISCLOSURE.md`](../DISCLOSURE.md) — material removed in commit `6475d07` and restored.
- [`protocol/DEVIATIONS.md`](../protocol/DEVIATIONS.md) — every departure from the analysis plan, with reasons.
- [`analysis/results/task0b_cohort_overlap.md`](../analysis/results/task0b_cohort_overlap.md) — cohort separation and within-cohort duplication.

## Known discrepancies between the manuscript and this repository

Each requires a manuscript edit; none is yet made.

| Manuscript states | Repository shows |
|---|---|
| 6 classification architectures | 13 |
| Seeds 42/123/2024, 18 runs | Seeds 67/1234/2025, 39 runs |
| 300x300 classification | 300 train / 224 inference (published); 300/300 (locked) |
| Table 3 threshold 0.502 | 0.5483 (manual) / 0.5000 (automatic) published; 0.5054 locked. `0.502635` is efficientnet_b4's threshold |
| nnU-Net three-fold | `fold_all` published; fold 0 with explicit split in the rebuild |
| U-Net 512x512 input | No `--target-size` argument exists; the published run trained at native resolution |
| Single software environment | Two, one of which no longer runs |
| DenseNet121 selected because of superior external performance | Reworded. Reported as post-selection multicentre evaluation; the internal-only ranking is a sensitivity analysis |
| 600 training patients | 600 images, 589 unique acquisitions |
