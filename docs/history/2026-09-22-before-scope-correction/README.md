# SPL-Models

Two-stage deep learning framework for automatic segmentation and benign-malignant differentiation of subpleural pulmonary lesions on grayscale ultrasound.

This repository contains the complete training, model-selection, inference and evaluation code, together with all training logs, random seeds, hyperparameters and pinned software environments.

Patient images, segmentation masks and labels are not redistributable.
See [`data/README.md`](data/README.md).
Trained model weights are distributed as release assets.

> **Status: revision complete.**
> The reported model is unchanged (DenseNet121).
> A checkpoint-selection defect was found in all three segmentation arms and corrected where it was consequential (see [Known defects](#known-defects-in-the-published-run)); nnU-Net's Dice changes by at most 0.011 on any evaluation cohort.
> [`docs/REPRODUCE.md`](docs/REPRODUCE.md) marks every artifact as CURRENT or SUPERSEDED.

## Start here

| Document | What it covers |
|---|---|
| [`docs/REPRODUCE.md`](docs/REPRODUCE.md) | Every reported number traced to the code, configuration and data that produced it |
| [`protocol/analysis_plan_v1.yaml`](protocol/analysis_plan_v1.yaml) | The pre-committed analysis plan for the locked re-analysis |
| [`protocol/DEVIATIONS.md`](protocol/DEVIATIONS.md) | Every departure from that plan, with reasons |

## The version-locked analysis pipeline

This section answers the requirement to "identify a single version-locked analysis pipeline" and to "explicitly mark any superseded code or descriptions as such".

**The version-locked pipeline is everything listed under ACTIVE below, governed by [`protocol/analysis_plan_v1.yaml`](protocol/analysis_plan_v1.yaml).**
That plan's SHA256 is recorded inside [`protocol/locked_pipeline.json`](protocol/locked_pipeline.json), so the lock record is cryptographically bound to the plan text it claims to follow.
Every reported number is produced by this set of scripts and by no other.
The per-artifact mapping is in [`docs/REPRODUCE.md`](docs/REPRODUCE.md); every departure from the plan is in [`protocol/DEVIATIONS.md`](protocol/DEVIATIONS.md).

### ACTIVE - the locked pipeline

| Stage | Script | Role |
|---|---|---|
| Plan | `protocol/analysis_plan_v1.yaml` | Pre-committed analysis plan: candidate pool, selection rule, tie-break, threshold rule, outcome contract |
| Enforcement | `protocol/gate.py` | Runtime hold-out. Blocks external images, labels and prior external results until the lock record exists |
| Enforcement | `protocol/test_gate.py` | 32 checks verifying the gate, including keyword-argument forms, `io.open`, subprocess and kernel-level blocking |
| Stage 1 | `seg-model-training/nnunet/prepare_dataset_with_validation.py` | Builds `Dataset001_lungval` with a genuine held-out split |
| Stage 1 | `seg-model-training/nnunet/preprocess_with_original_plans.sh` | Preprocesses using the published plans, so the validation split is the only changed variable |
| Stage 1 | `seg-model-training/nnunet/predict_all_cohorts.sh` | Predicts masks for all four cohorts from the retrained model |
| Stage 1 | `seg-model-training/unet_train_with_validation.py` | U-Net baseline with internal_val as the validation set |
| Stage 1 | `DeepLabV3Plus-Pytorch/main.py` (patched) | `--lung_val_img_dir` supplies a held-out validation cohort |
| Stage 2 | `protocol/run_protocol.py` | Phases C, D and E: internal-only training, lock, single external pass |
| Stage 2 | `analysis/code/export_locked_predictions.py` | Per-case probabilities for both mask variants from the locked model |
| Stage 2 | `analysis/code/calibrate_locked_model.py` | Post-hoc Platt calibration fitted on the tuning cohort only |
| Metrics | `analysis/code/seg_metrics_engine.py`, `make_table2.py` | Segmentation metrics and Table 2 |
| Metrics | `analysis/code/common.py` | Shared statistics: Wilson, bootstrap, DeLong, calibration, net benefit |
| Statistics | `analysis/code/task1_confidence_intervals.py`, `task2_threshold_policy.py`, `task3_calibration.py`, `task4_dca.py`, `task6_gt_vs_model_mask.py`, `subgroup_and_precision.py` | Reported tables and figures |
| Integrity | `analysis/code/task0_data_check.py`, `task0b_cohort_overlap.py` | Cohort integrity and cross-cohort image overlap |
| Audit | `analysis/code/summarize_training_logs.py` | Reconstructs checkpoint selection from the original logs. Evidence about the superseded run, and current as such |

The analysis layer is parameterised by `SPL_PRED_DIR`, `SPL_ARCH`, `SPL_RESULTS` and `SPL_FIGURES`, so the same statistical code produces both the locked and the superseded results with no duplicated statistics.
With no environment set it reproduces the superseded run exactly.

### SUPERSEDED - retained for audit, not used for any reported result

Nothing here is deleted.
Each file stays so the original analysis remains inspectable and the two can be compared.

| Script | Why superseded |
|---|---|
| `binary_classification/train.py` | Computed and logged External Test 1 and 2 AUC at every epoch for all 13 architectures. This is the code behind Reviewer Comment 1 |
| `binary_classification/infer_probs_tight.py` | The 224 px inference path. The locked pipeline uses 300 px at both training and inference. Also pointed at a directory that no longer existed |
| `binary_classification/run_tight_pipeline.sh` | Driver for the two scripts above |
| `binary_classification/heatmap.py` | Grad-CAM at 224 px. The script now takes the input size and checkpoint as parameters and reproduces the published overlays with no environment set |
| `analysis/code/train_locked.py` | Internal-only sweep, but ranked by mean accuracy across seeds rather than the declared rule. Its output `locked_selection_and_external.json` names a different model from the manuscript |
| `analysis/code/task5_model_selection.py` | **Cannot be run under the locked protocol.** It compares all 13 architectures on external data; the hold-out permits only the selected model to be scored externally. Replaced by the internal-only ranking in `protocol/locked_pipeline.json` |
| `analysis/code/run_seg_inference.py` | Regenerates masks from the superseded segmentation checkpoints |
| `analysis/code/evaluate_frozen_pipeline.py` | Frozen-pipeline evaluation of DenseNet121, written during the re-analysis and replaced by `export_locked_predictions.py` |
| `seg-model-training/benchmarking/*.py` | Three per-model evaluation scripts that do not share a metric definition. `seg_metrics_engine.py` supersedes all three |
| `seg-model-training/model_pipeline.sh` | Its U-Net and DeepLab hyperparameters contradict `README_SPL.md`, and both training invocations are commented out |
| `protocol/watchdog.sh` | Operational monitoring only. Not part of the analysis and produces no reported result |

### Superseded descriptions

These statements appear in the submitted manuscript or Supplement and are not supported by this repository.
Corrections are listed in [`docs/REPRODUCE.md`](docs/REPRODUCE.md#known-discrepancies-between-the-manuscript-and-this-repository).

| Superseded description | Status |
|---|---|
| "independent external validation" of Centers 2 and 3 | Withdrawn. Now post-selection multicentre performance evaluation |
| DenseNet121 selected because of superior external performance | Reworded. The selection is not presented as evidence of independence; Centers 2 and 3 are described as post-selection multicentre evaluation |
| nnU-Net "highest Dice across all cohorts" and therefore selected | Superseded. Selection is on the tuning cohort alone |
| 6 classification architectures | Superseded. 13 |
| Seeds 42/123/2024, 18 runs | Superseded. 67/1234/2025, 39 runs |
| 300x300 classification | Superseded. Published code used 300 train / 224 inference; locked pipeline uses 300 throughout |
| Table 3 threshold 0.502 | Superseded. Corresponds to neither reported model |
| nnU-Net three-fold | Superseded. Published run used `fold_all`; locked pipeline uses fold 0 with an explicit split |
| U-Net 512x512 input | Superseded. No such argument exists; the published run trained at native resolution |
| nnU-Net "Mean Validation Dice 0.9825" | Superseded. That is training Dice. Held-out Dice is 0.920 |
| Single software environment | Superseded. Two exist, one non-functional |

## Pipeline

The pipeline is sequential.
The two stages are trained and applied independently; there is no joint optimisation between them.

```
grayscale ultrasound image (one representative frame per patient)
    -> Stage 1  nnU-Net 2d           -> lesion mask
    -> lesion bounding box + 10% halo, resize, 2-channel input (image, mask)
    -> Stage 2  DenseNet121          -> malignancy probability
    -> fixed threshold (derived on the Center 1 tuning cohort) -> benign / malignant
```

Cohorts: 1059 images, one per patient.
Training 600 (Center 1), tuning 257 (Center 1), External Test 1 108 (Center 2), External Test 2 94 (Center 3).

The training cohort contains 600 images from **589 unique acquisitions**: eleven byte-identical duplicate pairs are present under distinct case identifiers, with consistent labels.
See [`analysis/results/task0b_cohort_overlap.md`](analysis/results/task0b_cohort_overlap.md).

## On the status of the external cohorts

The Center 2 and Center 3 results are reported as **post-selection multicentre performance evaluation**, not as independent external validation.

In the original development run, external metrics were computed and logged at every epoch for all thirteen candidate architectures.
Those logs are retained unedited in [`analysis/logs/training_logs/single_seed_67/`](analysis/logs/training_logs/single_seed_67/).
Model selection therefore took place in a setting where external performance was visible, and no contemporaneous record exists that fixes the architecture, preprocessing, hyperparameters, checkpoint policy or threshold before those results were examined.
No re-analysis can change that, because the limitation lies in the history of the data relative to the hypothesis rather than in the code.

What the locked re-analysis does establish, and what the repository supports:

- Architecture selection is made on Center 1 data alone, enforced at runtime rather than asserted, and recorded in [`protocol/gate_audit.jsonl`](protocol/gate_audit.jsonl).
- The external cohorts are read exactly once, after the pipeline is frozen and the lock record written.
- In the original run the retained checkpoint matched the internal-accuracy rule in **13 of 13** architectures, coincided with the external-AUC argmax in only 3 of 13 on Center 2 and 3 of 13 on Center 3, and forwent a mean of 0.045 and 0.053 external AUC respectively (maximum 0.271 and 0.281).

## Model selection

The reported model is **DenseNet121**, as in the original submission.
It was selected during development in a setting where external performance was visible, which is why the Center 2 and Center 3 results are described as post-selection multicentre performance evaluation rather than independent external validation.

As a sensitivity analysis, the same thirteen architectures were re-run using the Center 1 cohorts alone, with external data blocked at runtime.
Under an internal-data-only rule that analysis ranks EfficientNet-B0 first and DenseNet121 seventh across seeds, and no single architecture wins at all three seeds.
This supports the post-selection framing above: internal validation on 257 cases does not by itself identify DenseNet121.
It is reported as a sensitivity analysis, not as a change to the reported model.
Full ranking: [`protocol/locked_pipeline.json`](protocol/locked_pipeline.json).

## Repository layout

| Path | Contents |
|---|---|
| `protocol/` | Analysis plan, hold-out enforcement, phase runner, lock record, deviations |
| `binary_classification/` | Stage 2: original training, inference and Grad-CAM |
| `seg-model-training/` | Stage 1: training drivers and evaluation for nnU-Net, U-Net, DeepLabv3+ |
| `analysis/code/` | Evaluation and statistical analysis scripts |
| `analysis/results/` | Generated result tables (CSV, JSON, markdown) |
| `analysis/figures/` | Calibration and decision-curve figures (PDF, PNG) |
| `analysis/logs/training_logs/` | Every training log, both runs, unedited |
| `test/` | Exploratory evaluation on a cohort outside this study; not used for any reported result |
| `docs/` | Result-to-code map |

## Configuration

### Random seeds

| Purpose | Seed |
|---|---|
| Primary run | 67 |
| Robustness repeats, not used for selection | 1234, 2025 |
| All bootstrap procedures | 20260904, 2000 resamples |

`torch.backends.cudnn.deterministic` is `False`, so runs are reproducible in distribution rather than bitwise.

### Stage 1, segmentation

All three arms are trained on the Center 1 training cohort (n = 600) and validated on the Center 1 tuning cohort (n = 257).
This is a correction; see [Known defects](#known-defects-in-the-published-run).

nnU-Net v2 (`nnunetv2==2.8.1`), dataset `Dataset001_lungval`, configuration `2d`, `nnUNetTrainer` / `nnUNetPlans`, **fold 0 with an explicit `splits_final.json`**.
Patch 896 x 1792, batch 2, spacing 1.0 x 1.0, `CTNormalization`, `use_mask_for_norm=False`, 1000 epochs, SGD, initial LR 0.01, weight decay 3e-05, foreground oversampling 0.33.
The plans are copied verbatim from the published `Dataset000_lung` so the architecture configuration is unchanged and normalisation statistics remain derived from the training cohort alone.

U-Net (Pytorch-UNet): `n_channels=1`, `n_classes=2`, `bilinear=False`, native input resolution, per-image max scaling, 350 epochs, batch 1, RMSprop, LR 5e-4.

DeepLabv3+ (DeepLabV3Plus-Pytorch): `deeplabv3plus_mobilenet`, output stride 16, 2 classes, 512 x 512 crop, batch 4, 30000 iterations, LR 0.01 with poly schedule, ImageNet normalisation.

### Stage 2, classification

Input 2 channels (image, mask), lesion bounding box + 10% halo (`HALO_FRAC=0.10`), per-channel standardisation.
Input size **300 x 300 at both training and inference**.
40 epochs, batch 16, AdamW, LR 1e-4, weight decay 5e-4, `CosineAnnealingLR` with `eta_min=1e-6`, early stopping patience 8 on internal-validation AUC.
`BCEWithLogitsLoss` weighted per sample by `mask.mean()*0.9 + 0.1` clamped at 0.1, label smoothing 0.1, `WeightedRandomSampler` on inverse class frequency.
ImageNet pretraining, first convolution adapted to 2 channels by averaging the RGB filters.

Augmentation, training only: horizontal flip p=0.5; rotation p=0.5, uniform -20 to +20 degrees, reflect padding; gamma p=0.7, uniform 0.6 to 1.4; contrast p=0.7, uniform 0.75 to 1.25; Gaussian noise p=0.5, SD 6.0 on a 0-255 scale.

Architectures evaluated: Inception-v3, VGG-19, ResNet-18/50/101, EfficientNet-B0 through B5, DenseNet121, DenseNet201.
All thirteen are run; the candidate pool is not narrowed.

### Threshold

The operating threshold is derived on the Center 1 tuning cohort only, by Youden's J, and applied unchanged to both external cohorts.
For DenseNet121 the value is **0.5483** with manual masks and **0.5000** with automatic masks, as originally reported.
`0.502635`, which appeared as the threshold in Table 3 of the submission, is EfficientNet-B4's tuning threshold and corresponds to no reported model; it is corrected.

## Known defects in the published run

Each of these is corrected in the locked re-analysis and recorded in [`protocol/DEVIATIONS.md`](protocol/DEVIATIONS.md).

1. **All three segmentation arms selected their checkpoint on training data.**
   nnU-Net `fold_all` sets `val_keys = tr_keys` (`nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:565-569`), so `fold_all/validation/` holds exactly the 600 training cases and the reported "Mean Validation Dice 0.9825" is training Dice.
   Pytorch-UNet `--validation 0` uses the training set as the validation set; its log header reads `val_dice(train)`.
   DeepLabv3+ builds train and validation from the same `all_indices`.
   Metric values in Table 2 were computed per cohort and are not themselves invalid; the checkpoint selection behind them was.

2. **Training and inference ran at different resolutions**, 300 and 224.
   The locked pipeline uses 300 throughout.

3. **`cv2.warpAffine` on an `(H, W, 1)` array returns `(H, W)`**, so the subsequent `img[..., 0]` indexed a single image column and intensity augmentation affected one column in rotated samples.
   Corrected in `protocol/run_protocol.py::LockedDataset`.

4. **Early stopping on 257 tuning cases is unstable.**
   Retained epochs range from 1 to 16 across the sweep, and a single architecture's internal AUC varies by up to 0.07 across seeds.
   This is a limitation of the cohort size, not of the implementation.

5. **`binary_classification/infer_probs_tight.py` pointed at a directory that no longer exists** (`new_data/`), so it could not run as deposited.
   The path is corrected to `data/`; the two are the same images, verified by reproducing all 918 committed predictions to within 2.3e-4.

6. **Segmentation metrics are computed on the lesion class only, at native resolution.**
   `analysis/code/seg_metrics_engine.py` is the reference implementation and supersedes the three per-model benchmark scripts, which did not share a metric definition.

## Environment

Two environments exist.
Both are declared, because the repository has always contained two.

| Environment | Used for | Status |
|---|---|---|
| `~/venvs/prism` | The locked re-analysis, all stages | Working. Python 3.11.14, torch 2.13.0+cu126, torchvision 0.28.0+cu126, CUDA 12.6, nnunetv2 2.8.1, numpy 1.26.4, scipy 1.10.0, scikit-learn 1.9.0, nibabel 5.4.2, SimpleITK 2.5.6, opencv-python 4.11.0.86 |
| `nnunet-env` | The published nnU-Net run (torch 2.9.1+cu128) | Non-functional. Created at a different path and later moved, so console-script shebangs point at a missing interpreter and `import nnunetv2` fails. The published nnU-Net run cannot be reproduced in the environment that produced it. |

Single NVIDIA GeForce RTX 4090.
Complete pinned environment: [`analysis/environment_lock.txt`](analysis/environment_lock.txt).

Third-party frameworks are not vendored.
Install from upstream at the pinned versions: [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet), [DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch).
Local modifications to the upstream trees are described in [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Reproducing

Full commands, in order, with the status of each output: [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

```bash
PY=~/venvs/prism/bin/python

# Stage 1: build a dataset with a held-out validation split, then train
$PY seg-model-training/nnunet/prepare_dataset_with_validation.py
bash seg-model-training/nnunet/preprocess_with_original_plans.sh
nnUNetv2_train 1 2d 0
$PY seg-model-training/unet_train_with_validation.py

# Stage 2: internal-only training, lock, then a single external pass
$PY protocol/run_protocol.py train
$PY protocol/run_protocol.py lock
$PY protocol/run_protocol.py external

# Evaluation and statistics
$PY analysis/code/seg_metrics_engine.py
$PY analysis/code/make_table2.py
bash analysis/code/run_all.sh

# Verification
$PY protocol/test_gate.py                      # hold-out enforcement, 32 checks
$PY analysis/code/task0b_cohort_overlap.py     # cross-cohort image overlap
$PY analysis/code/verify_cached_dataset.py     # cached loader equivalence
```

## Training logs

Every training log from both runs is included unedited under [`analysis/logs/training_logs/`](analysis/logs/training_logs/): 13 logs from the original run (seed 67) and 39 from the multi-seed sweep.

`analysis/code/summarize_training_logs.py` recovers the full per-epoch series from each log and reports which epoch was retained against which epoch each candidate selection rule would have chosen.

## License

MIT. See [`LICENSE`](LICENSE).
