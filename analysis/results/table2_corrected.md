# Table 2 (corrected) - Segmentation performance

All metrics are computed on the LESION (foreground) class only, at native image
resolution, from the locked predicted masks, using one shared metric engine
(`analysis/code/seg_metrics_engine.py`).
Values are mean (95% patient-level bootstrap percentile CI, 2000 resamples,
seed 20260904), except HD95 and ASSD which are median (IQR) because their
distributions are right-skewed.

Explicit denominators:

- Precision = TP / (TP + FP), denominator = all pixels PREDICTED lesion.
- Recall = TP / (TP + FN), denominator = all pixels TRULY lesion.
- FPR = FP / (FP + TN), denominator = all pixels TRULY background.
- HD95 and ASSD are in pixels on the native grid, undefined when either mask is empty.

## Training (model fit) (n = 600)

| Model | Dice | IoU | Precision | Recall | FPR | HD95_median_IQR_px | ASSD_median_IQR_px | Empty_predictions |
|---|---|---|---|---|---|---|---|---|
| nnU-Net | 0.982 (0.982-0.983) | 0.966 (0.964-0.967) | 0.982 (0.981-0.983) | 0.983 (0.982-0.984) | 0.00066 (0.00062-0.00070) | 4.0 (3.2-5.0) | 1.5 (1.2-1.8) | 0 |
| DeepLabv3+ | 0.721 (0.700-0.742) | 0.619 (0.597-0.639) | 0.875 (0.862-0.888) | 0.695 (0.671-0.717) | 0.00439 (0.00387-0.00499) | 49.4 (30.0-89.0) | 17.1 (10.0-28.5) | 27 |
| U-Net | 0.741 (0.726-0.756) | 0.617 (0.601-0.633) | 0.758 (0.741-0.775) | 0.779 (0.762-0.796) | 0.00974 (0.00894-0.01060) | 83.2 (46.6-137.3) | 25.8 (15.4-42.9) | 3 |

## Development/tuning (Center 1) (n = 257)

| Model | Dice | IoU | Precision | Recall | FPR | HD95_median_IQR_px | ASSD_median_IQR_px | Empty_predictions |
|---|---|---|---|---|---|---|---|---|
| nnU-Net | 0.920 (0.911-0.928) | 0.858 (0.847-0.869) | 0.925 (0.914-0.935) | 0.925 (0.917-0.934) | 0.00410 (0.00329-0.00504) | 23.0 (14.9-39.2) | 6.9 (5.0-10.4) | 0 |
| DeepLabv3+ | 0.766 (0.737-0.794) | 0.664 (0.635-0.693) | 0.907 (0.893-0.921) | 0.731 (0.697-0.763) | 0.00408 (0.00338-0.00483) | 55.5 (29.9-97.0) | 17.1 (10.4-28.2) | 7 |
| U-Net | 0.797 (0.781-0.812) | 0.679 (0.660-0.698) | 0.823 (0.804-0.842) | 0.814 (0.793-0.834) | 0.00841 (0.00731-0.00957) | 83.1 (43.8-142.4) | 23.3 (14.7-41.6) | 0 |

## External Test 1 (Center 2) (n = 108)

| Model | Dice | IoU | Precision | Recall | FPR | HD95_median_IQR_px | ASSD_median_IQR_px | Empty_predictions |
|---|---|---|---|---|---|---|---|---|
| nnU-Net | 0.913 (0.898-0.926) | 0.848 (0.826-0.868) | 0.918 (0.900-0.934) | 0.920 (0.903-0.934) | 0.00284 (0.00231-0.00343) | 18.5 (12.3-32.9) | 5.8 (4.1-9.1) | 0 |
| DeepLabv3+ | 0.783 (0.736-0.825) | 0.693 (0.647-0.737) | 0.918 (0.899-0.935) | 0.746 (0.694-0.793) | 0.00317 (0.00246-0.00391) | 34.2 (22.8-58.7) | 10.9 (7.3-19.1) | 3 |
| U-Net | 0.765 (0.729-0.796) | 0.648 (0.609-0.683) | 0.760 (0.720-0.798) | 0.828 (0.792-0.859) | 0.01027 (0.00843-0.01221) | 75.2 (32.5-130.7) | 23.9 (12.5-36.2) | 1 |

## External Test 2 (Center 3) (n = 94)

| Model | Dice | IoU | Precision | Recall | FPR | HD95_median_IQR_px | ASSD_median_IQR_px | Empty_predictions |
|---|---|---|---|---|---|---|---|---|
| nnU-Net | 0.915 (0.901-0.926) | 0.848 (0.827-0.866) | 0.930 (0.911-0.945) | 0.910 (0.894-0.924) | 0.00407 (0.00311-0.00517) | 25.9 (16.5-42.4) | 7.7 (5.5-11.2) | 0 |
| DeepLabv3+ | 0.736 (0.678-0.788) | 0.639 (0.582-0.692) | 0.904 (0.871-0.930) | 0.694 (0.633-0.750) | 0.00361 (0.00270-0.00460) | 55.7 (31.7-89.9) | 19.3 (10.3-31.9) | 2 |
| U-Net | 0.780 (0.748-0.807) | 0.660 (0.625-0.691) | 0.827 (0.792-0.859) | 0.788 (0.751-0.823) | 0.00895 (0.00710-0.01100) | 99.1 (59.3-143.8) | 28.2 (18.2-43.1) | 0 |
