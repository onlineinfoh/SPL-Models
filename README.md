# SPL-Models

Two-stage deep learning framework for automatic segmentation and benign-malignant differentiation of subpleural pulmonary lesions on grayscale ultrasound.

This repository contains the complete training, model-selection, inference and evaluation code, together with all training logs, random seeds, hyperparameters and a pinned software environment.

Patient images, segmentation masks and labels are not redistributable. See [`data/README.md`](data/README.md).
Trained model weights are available from the authors on reasonable request.

**Where to start.** [`docs/REPRODUCE.md`](docs/REPRODUCE.md) maps every reported table to the exact script, configuration, seed, mask variant and inference resolution that produced it, and lists the known gaps.
[`docs/CHANGES_AND_REMOVALS.md`](docs/CHANGES_AND_REMOVALS.md) records everything changed or removed during revision, and why.

## Pipeline

The pipeline is sequential. The two stages are trained and applied independently; there is no joint optimisation between them.

```
grayscale ultrasound image (one representative frame per patient)
    -> Stage 1  nnU-Net 2d           -> lesion mask
    -> lesion bounding box + 10% halo, resize, 2-channel input (image, mask)
    -> Stage 2  DenseNet121          -> malignancy probability
    -> fixed threshold (derived on the Center 1 tuning cohort) -> benign / malignant
```

Cohorts: 1059 patients, one image analysed per patient. Training 600, tuning 257 (Center 1), External Test 1 108 (Center 2), External Test 2 94 (Center 3).
11 duplicate images were found within the training cohort; there was no overlap with either external cohort. See [`docs/CHANGES_AND_REMOVALS.md`](docs/CHANGES_AND_REMOVALS.md).

## Repository layout

| Path | Contents |
|---|---|
| `docs/` | Result-to-code map, revision record, Table 2 FPR derivation |
| `binary_classification/` | Stage 2: training, model selection, inference, Grad-CAM |
| `seg-model-training/` | Stage 1: training driver and evaluation for nnU-Net, U-Net, DeepLabv3+ |
| `analysis/code/` | Evaluation and statistical analysis scripts |
| `analysis/results/` | Generated result tables (CSV, JSON, markdown) |
| `analysis/figures/` | Calibration and decision-curve figures (PDF, PNG) |
| `analysis/logs/training_logs/` | Every training log, both runs, unedited |
| `analysis/runs_locked/` | Per-run logs from the multi-seed sweep |

## Scripts

Stage 1, segmentation:

| Script | Purpose |
|---|---|
| `seg-model-training/model_pipeline.sh` | Training driver |
| `seg-model-training/benchmarking/nnunet_benchmarking.py` | nnU-Net evaluation |
| `seg-model-training/benchmarking/unet_benchmark.py` | U-Net evaluation |
| `seg-model-training/benchmarking/deeplab_benchmark.py` | DeepLabv3+ evaluation |
| `seg-model-training/benchmarking/visual_check_overlay.py` | Overlay visualisation |

Stage 2, classification:

| Script | Purpose |
|---|---|
| `binary_classification/build_labels.py` | Label construction from cohort spreadsheets |
| `binary_classification/check_data.py` | Dataset integrity checks |
| `binary_classification/train.py` | Training across all candidate architectures |
| `binary_classification/infer_probs_tight.py` | Inference, threshold selection, reported metrics |
| `binary_classification/heatmap.py` | Grad-CAM maps |
| `binary_classification/run_tight_pipeline.sh` | End-to-end driver |

Evaluation and analysis:

| Script | Purpose |
|---|---|
| `analysis/code/seg_metrics_engine.py` | Segmentation metrics: Dice, IoU, precision, recall, FPR, HD95, ASSD, bootstrap CIs |
| `analysis/code/run_seg_inference.py` | Regenerates predicted masks at native resolution |
| `analysis/code/make_table2.py` | Segmentation results table |
| `analysis/code/make_table2b.py` | Boundary-metric table (HD95, ASSD) |
| `analysis/code/task1_confidence_intervals.py` | Classification metrics and confidence intervals |
| `analysis/code/task2_threshold_policy.py` | Threshold-selection comparison |
| `analysis/code/task3_calibration.py` | Calibration intercept, slope, Brier score, curves |
| `analysis/code/task4_dca.py` | Decision-curve analysis |
| `analysis/code/task5_model_selection.py` | Architecture ranking, DeLong tests |
| `analysis/code/task6_gt_vs_model_mask.py` | Manual versus automatic mask comparison |
| `analysis/code/train_locked.py` | Multi-seed training sweep, internal data only |
| `analysis/code/summarize_training_logs.py` | Recovers per-epoch series and checkpoint selection from logs |
| `analysis/code/verify_cached_dataset.py` | Asserts the cached loader is bit-identical to the original |

## Configuration

### Random seeds

| Purpose | Seed |
|---|---|
| Training and inference | 67 |
| Multi-seed sweep | 67, 1234, 2025 |
| All bootstrap procedures | 20260904, 2000 resamples |

`torch.backends.cudnn.deterministic` is `False`, so runs are reproducible in distribution rather than bitwise.

### Stage 1, nnU-Net (selected)

nnU-Net v2 (`nnunetv2==2.8.1`), `Dataset000_lung`, configuration `2d`, `nnUNetTrainer` / `nnUNetPlans`, fold `all`, `checkpoint_best.pth`.
Patch 896 x 1792, batch 2, spacing 1.0 x 1.0, `CTNormalization`, `use_mask_for_norm=False`, 1000 epochs, SGD, initial LR 0.01, weight decay 3e-05, foreground oversampling 0.33.

### Stage 1 baselines

U-Net (Pytorch-UNet): `n_channels=1`, `n_classes=2`, `bilinear=False`, 512 x 512 input, per-image max scaling.
350 epochs, batch 1, validation split 0, initial LR 5e-4.
Optimiser `RMSprop`, scheduler `ReduceLROnPlateau` on max Dice with patience 5, both upstream defaults
(`Pytorch-UNet/train.py:115` and `:117`).

DeepLabv3+ (DeepLabV3Plus-Pytorch): `deeplabv3plus_mobilenet`, output stride 16, 2 classes, 512 x 512 input, ImageNet normalisation.
30000 iterations, batch 4, crop 512, initial LR 0.01, validation interval 200.
Optimiser `SGD` with momentum 0.9, scheduler `PolyLR` with power 0.9, both upstream defaults
(`DeepLabV3Plus-Pytorch/main.py:321` and `:328`).

Neither baseline uses Adam, and neither uses cosine annealing. Cosine annealing is used only in
Stage 2 (`train.py:602`); nnU-Net uses its own SGD polynomial schedule.

Both baselines scale intensity by the per-image maximum, `img = img / img.max()`
(`benchmarking/unet_benchmark.py:137-138`, `benchmarking/deeplab_benchmark.py:127-128`).
There is no percentile clipping and no minimum subtraction in either baseline path.
DeepLabv3+ additionally applies ImageNet mean and standard deviation after that scaling.
Only nnU-Net performs percentile clipping, as part of `CTNormalization`, which clips to the dataset
0.5th and 99.5th percentiles and then z-scores using dataset-level statistics.

### Stage 2, classification

Input 2 channels (image, mask), lesion bounding box + 10% halo (`HALO_FRAC=0.10`), per-channel standardisation.
Input size 300 x 300 at training (`train.py`), 224 x 224 at inference (`infer_probs_tight.py`).
40 epochs, batch 16, AdamW, LR 1e-4, weight decay 5e-4, `CosineAnnealingLR` with `eta_min=1e-6`, early stopping patience 8 on tuning-set accuracy.
`BCEWithLogitsLoss` weighted per sample by `mask.mean()*0.9 + 0.1` clamped at 0.1, label smoothing 0.1, `WeightedRandomSampler` on inverse class frequency.
Dropout 0.4 before the single-logit head on every architecture.
ImageNet pretraining, first convolution adapted to 2 channels by averaging the RGB filters across the input dimension and copying that averaged kernel into both input channels (`_adapt_first_conv`).

Grad-CAM (`heatmap.py`) runs at the same 224 x 224 inference resolution, then resizes the map to the native lesion crop for overlay.

Augmentation, training only: horizontal flip p=0.5; rotation p=0.5, uniform -20 to +20 degrees, reflect padding; gamma p=0.7, uniform 0.6 to 1.4; contrast p=0.7, uniform 0.75 to 1.25; Gaussian noise p=0.5, SD 6.0 on a 0-255 scale.

Architectures evaluated: Inception-v3, VGG-19, ResNet-18/50/101, EfficientNet-B0 through B5, DenseNet121, DenseNet201.

### Threshold

The operating threshold is selected in `infer_probs_tight.py` (`_best_threshold_from_rows`) on the Center 1 tuning cohort only, as the probability threshold maximising tuning-set accuracy, then applied unchanged to both external cohorts.
For DenseNet121 the value is 0.5483 with manual masks and 0.5000 with automatic masks. For this model that threshold also coincides with the Youden-optimal point.
`train.py` uses a fixed 0.5 for the metrics it prints during training; those are monitoring output and are not the reported results.

## Implementation notes

1. Training and inference use different input resolutions, 300 and 224. Reported results come from the 224 inference harness.
2. If a segmentation mask contains no positive pixel, a centred square crop replaces the lesion bounding box. Across all 1059 study cases this branch was never taken.
3. Segmentation metrics are computed on the lesion class only, at native resolution. `analysis/code/seg_metrics_engine.py` is the reference implementation and supersedes the three per-model benchmark scripts, which did not share a metric definition.
4. In `train.py`, `cv2.warpAffine` on an `(H, W, 1)` array returns `(H, W)`. After a rotation the subsequent `img[..., 0]` therefore indexes a single image column, so the intensity augmentations affect one column in samples where rotation is applied.
5. Early stopping with patience 8 on 257 tuning cases is unstable: across the 39 sweep runs the retained epoch has median 4 and range 1 to 12, with 13 of 39 runs retaining an epoch 1 or 2 checkpoint.
   Those 39 runs are 13 classification architectures x 3 seeds. No segmentation model is among them.
6. The three selection steps do not all use the same mask variant. The architecture ranking and the operating threshold are computed on the automatic-mask (`model`) variant in `infer_probs_tight.py`, matching deployment. The retained checkpoint is chosen on tuning-set accuracy computed with the manual masks, because `train.py` loads `data/val/seg_v` only and has no automatic-mask path. All three read the Center 1 tuning cohort and nothing else; `analysis/results/checkpoint_selection_summary.json` records that the retained epoch equals the internal-validation argmax for 13 of 13 architectures and coincides with the external argmax for only 3.
7. Reported results are seed 67 throughout, the `--seeds` default at `train.py:566`. They are not a per-architecture best-of-three-seeds selection: across the sweep, seed 67 is the highest-tuning-accuracy seed for 5 of the 13 architectures.

## Environment

Python 3.11.14, torch 2.13.0+cu126, torchvision 0.28.0+cu126, CUDA 12.6, cuDNN 91002, nnunetv2 2.8.1, numpy 1.26.4, scipy 1.10.0, scikit-learn 1.9.0, nibabel 5.4.2, SimpleITK 2.5.6, opencv-python 4.11.0.86.
Single NVIDIA GeForce RTX 4090.

Complete pinned environment: [`analysis/environment_lock.txt`](analysis/environment_lock.txt).

Third-party frameworks are not vendored. Install from upstream at the pinned versions:
[nnU-Net](https://github.com/MIC-DKFZ/nnUNet), [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet), [DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch).

## Reproducing

Create the environment from [`analysis/environment_lock.txt`](analysis/environment_lock.txt) first, then point `PY` at its interpreter.

```bash
PY="${PY:-python3}"

# Stage 1: predicted masks, then metrics
$PY analysis/code/run_seg_inference.py
$PY analysis/code/seg_metrics_engine.py
$PY analysis/code/make_table2.py
$PY analysis/code/make_table2b.py

# Stage 2: training, then inference with the tuning-set threshold
$PY binary_classification/train.py
$PY binary_classification/infer_probs_tight.py

# Evaluation and statistics
bash analysis/code/run_all.sh

# Multi-seed sweep, internal data only
$PY analysis/code/verify_cached_dataset.py
$PY analysis/code/train_locked.py --seeds 67 1234 2025

# Checkpoint-selection audit over the training logs
$PY analysis/code/summarize_training_logs.py
```

## Training logs

Every training log from both runs is included unedited under [`analysis/logs/training_logs/`](analysis/logs/training_logs/):
13 logs from the submitted run (seed 67), and 39 logs from the multi-seed sweep.

`analysis/code/summarize_training_logs.py` recovers the full per-epoch series from each log and reports which epoch was retained against which epoch each candidate selection rule would have chosen.

## Not in this repository

Four reported results have no producer here. They are listed with the rest of the known gaps in
[`docs/REPRODUCE.md`](docs/REPRODUCE.md#known-gaps), gaps 9 to 13.

| Reported result | Status |
|---|---|
| Table 4, Supplementary Table 8, Figure 6, reader study | No reader responses and no analysis code. No case-cluster bootstrap or McNemar implementation exists in this tree. |
| Table 1, baseline characteristics | No ANOVA, chi-square or Fisher exact routine exists in this tree. |
| Figure 3, ROC curves | Not generated here. The `roc.png` files under `runs/` are 300 px training-time monitoring plots, not Figure 3. |
| Interobserver Dice | Requires the two pre-adjudication annotation sets, which are not deposited. |

## Citation

This repository accompanies the manuscript:

> A Grayscale Ultrasound-Based Two-Stage Deep Learning Framework for Automatic Segmentation and Benign-Malignant Differentiation of Subpleural Pulmonary Lesions.
> Submitted to *BMC Medical Imaging*.

Repository: <https://github.com/onlineinfoh/SPL-Models>

The author list, volume and DOI are added on acceptance.

## License

MIT. See [`LICENSE`](LICENSE).
