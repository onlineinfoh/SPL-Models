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
| nnU-Net | 0.959 (0.957-0.961) | 0.922 (0.919-0.926) | 0.957 (0.954-0.960) | 0.962 (0.960-0.964) | 0.00153 (0.00142-0.00166) | 9.8 (7.6-12.7) | 3.5 (2.8-4.4) | 0 |
| DeepLabv3+ | 0.636 (0.612-0.659) | 0.527 (0.505-0.548) | 0.824 (0.808-0.840) | 0.620 (0.594-0.645) | 0.00703 (0.00621-0.00789) | 67.1 (45.2-108.5) | 23.2 (15.0-38.1) | 51 |
| U-Net | 0.741 (0.726-0.756) | 0.617 (0.601-0.633) | 0.758 (0.741-0.775) | 0.779 (0.762-0.796) | 0.00974 (0.00894-0.01060) | 83.2 (46.6-137.3) | 25.8 (15.4-42.9) | 3 |

## Development/tuning (Center 1) (n = 257)

| Model | Dice | IoU | Precision | Recall | FPR | HD95_median_IQR_px | ASSD_median_IQR_px | Empty_predictions |
|---|---|---|---|---|---|---|---|---|
| nnU-Net | 0.917 (0.907-0.925) | 0.853 (0.841-0.864) | 0.922 (0.911-0.931) | 0.922 (0.913-0.932) | 0.00386 (0.00314-0.00477) | 23.9 (14.9-41.1) | 7.2 (5.4-10.9) | 0 |
| DeepLabv3+ | 0.732 (0.701-0.762) | 0.626 (0.596-0.654) | 0.892 (0.877-0.907) | 0.699 (0.663-0.733) | 0.00520 (0.00433-0.00610) | 60.1 (38.0-99.1) | 19.6 (13.2-30.7) | 7 |
| U-Net | 0.797 (0.781-0.812) | 0.679 (0.660-0.698) | 0.823 (0.804-0.842) | 0.814 (0.793-0.834) | 0.00841 (0.00731-0.00957) | 83.1 (43.8-142.4) | 23.3 (14.7-41.6) | 0 |

## External Test 1 (Center 2) (n = 108)

| Model | Dice | IoU | Precision | Recall | FPR | HD95_median_IQR_px | ASSD_median_IQR_px | Empty_predictions |
|---|---|---|---|---|---|---|---|---|
| nnU-Net | 0.908 (0.892-0.922) | 0.841 (0.817-0.861) | 0.916 (0.897-0.934) | 0.913 (0.893-0.929) | 0.00274 (0.00227-0.00326) | 19.4 (12.4-32.8) | 6.0 (4.1-10.4) | 0 |
| DeepLabv3+ | 0.718 (0.660-0.770) | 0.623 (0.566-0.672) | 0.830 (0.789-0.867) | 0.721 (0.659-0.777) | 0.00663 (0.00534-0.00804) | 46.3 (31.5-82.6) | 14.6 (10.7-28.8) | 4 |
| U-Net | 0.765 (0.729-0.796) | 0.648 (0.609-0.683) | 0.760 (0.720-0.798) | 0.828 (0.792-0.859) | 0.01027 (0.00843-0.01221) | 75.2 (32.5-130.7) | 23.9 (12.5-36.2) | 1 |

## External Test 2 (Center 3) (n = 94)

| Model | Dice | IoU | Precision | Recall | FPR | HD95_median_IQR_px | ASSD_median_IQR_px | Empty_predictions |
|---|---|---|---|---|---|---|---|---|
| nnU-Net | 0.904 (0.886-0.918) | 0.833 (0.808-0.854) | 0.924 (0.903-0.942) | 0.898 (0.879-0.915) | 0.00389 (0.00304-0.00486) | 29.6 (18.9-46.9) | 8.9 (5.6-13.8) | 0 |
| DeepLabv3+ | 0.728 (0.678-0.774) | 0.617 (0.568-0.663) | 0.857 (0.820-0.889) | 0.703 (0.645-0.755) | 0.00766 (0.00612-0.00930) | 73.0 (42.4-106.1) | 25.2 (16.1-34.5) | 1 |
| U-Net | 0.780 (0.748-0.807) | 0.660 (0.625-0.691) | 0.827 (0.792-0.859) | 0.788 (0.751-0.823) | 0.00895 (0.00710-0.01100) | 99.1 (59.3-143.8) | 28.2 (18.2-43.1) | 0 |
