# SPL-Models

Two-stage deep learning framework for automatic segmentation and benign-malignant differentiation of subpleural pulmonary lesions on grayscale ultrasound.

This repository contains the complete training, model-selection, inference and evaluation code, together with all training logs, random seeds, hyperparameters and a pinned software environment.

Patient images, segmentation masks and labels are not redistributable. See [`data/README.md`](data/README.md).
Trained model weights are available from the authors on reasonable request.

## Pipeline

The pipeline is sequential. The two stages are trained and applied independently; there is no joint optimisation between them.

```
grayscale ultrasound image (one representative frame per patient)
    -> Stage 1  nnU-Net 2d           -> lesion mask
    -> lesion bounding box + 10% halo, resize, 2-channel input (image, mask)
    -> Stage 2  DenseNet121          -> malignancy probability
    -> fixed threshold (derived on the Center 1 tuning cohort) -> benign / malignant
```

Cohorts: 1059 patients, one image per patient. Training 600, tuning 257 (Center 1), External Test 1 108 (Center 2), External Test 2 94 (Center 3).

## Repository layout

| Path | Contents |
|---|---|
| `binary_classification/` | Stage 2: training, model selection, inference, Grad-CAM |
| `seg-model-training/` | Stage 1: training driver and evaluation for nnU-Net, U-Net, DeepLabv3+ |
| `analysis/code/` | Evaluation and statistical analysis scripts |
| `analysis/results/` | Generated result tables (CSV, JSON, markdown) |
| `analysis/figures/` | Calibration and decision-curve figures (PDF, PNG) |
| `analysis/logs/training_logs/` | Every training log, both runs, unedited |
| `analysis/runs_locked/` | Per-run logs from the multi-seed sweep |
| `test/` | Standalone evaluation utilities |

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
DeepLabv3+ (DeepLabV3Plus-Pytorch): `deeplabv3plus_mobilenet`, output stride 16, 2 classes, 512 x 512 input, ImageNet normalisation.

### Stage 2, classification

Input 2 channels (image, mask), lesion bounding box + 10% halo (`HALO_FRAC=0.10`), per-channel standardisation.
Input size 300 x 300 at training (`train.py`), 224 x 224 at inference (`infer_probs_tight.py`).
40 epochs, batch 16, AdamW, LR 1e-4, weight decay 5e-4, `CosineAnnealingLR` with `eta_min=1e-6`, early stopping patience 8 on tuning-set accuracy.
`BCEWithLogitsLoss` weighted per sample by `mask.mean()*0.9 + 0.1` clamped at 0.1, label smoothing 0.1, `WeightedRandomSampler` on inverse class frequency.
ImageNet pretraining, first convolution adapted to 2 channels by averaging the RGB filters.

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

## Environment

Python 3.11.14, torch 2.13.0+cu126, torchvision 0.28.0+cu126, CUDA 12.6, cuDNN 91002, nnunetv2 2.8.1, numpy 1.26.4, scipy 1.10.0, scikit-learn 1.9.0, nibabel 5.4.2, SimpleITK 2.5.6, opencv-python 4.11.0.86.
Single NVIDIA GeForce RTX 4090.

Complete pinned environment: [`analysis/environment_lock.txt`](analysis/environment_lock.txt).

Third-party frameworks are not vendored. Install from upstream at the pinned versions:
[nnU-Net](https://github.com/MIC-DKFZ/nnUNet), [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet), [DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch).

## Reproducing

```bash
PY=~/venvs/prism/bin/python

# Stage 1: predicted masks, then metrics
$PY analysis/code/run_seg_inference.py
$PY analysis/code/seg_metrics_engine.py
$PY analysis/code/make_table2.py

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

## License

MIT. See [`LICENSE`](LICENSE).
