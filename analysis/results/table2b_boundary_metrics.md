# Table 2b - Boundary-sensitive segmentation metrics

HD95 = 95th percentile of the symmetric boundary distance set.
ASSD = mean of the symmetric boundary distance set.
Both are in **pixels on the native image grid** (median image ~1.25-1.5 megapixels).
Both are undefined when either mask is empty; `n_undefined` counts those cases and they are excluded.
Distributions are strongly right-skewed, so **median (IQR) is the primary summary**;
the mean is given with a 95 percent patient-level bootstrap CI (2000 resamples, seed 20260904).

## Development/tuning (Center 1)

| Model | Metric | n | Median (IQR) px | Mean (95% CI) px | p95 px | Undefined |
|---|---|---|---|---|---|---|
| nnU-Net | HD95 | 257 | 23.0 (14.87-39.18) | 35.65 (31.01-40.91) | 110.89 | 0 |
| nnU-Net | ASSD | 257 | 6.89 (5.01-10.41) | 9.81 (8.47-11.67) | 22.28 | 0 |
| DeepLabv3+ | HD95 | 250 | 55.47 (29.90-96.97) | 75.23 (67.50-83.75) | 205.7 | 7 |
| DeepLabv3+ | ASSD | 250 | 17.08 (10.41-28.20) | 24.44 (21.55-28.04) | 62.42 | 7 |
| U-Net | HD95 | 257 | 83.1 (43.84-142.39) | 100.93 (92.32-109.60) | 236.9 | 0 |
| U-Net | ASSD | 257 | 23.27 (14.70-41.64) | 30.85 (27.98-33.67) | 73.62 | 0 |

## External Test 1 (Center 2)

| Model | Metric | n | Median (IQR) px | Mean (95% CI) px | p95 px | Undefined |
|---|---|---|---|---|---|---|
| nnU-Net | HD95 | 108 | 18.53 (12.30-32.90) | 30.35 (23.98-38.70) | 98.85 | 0 |
| nnU-Net | ASSD | 108 | 5.85 (4.14-9.13) | 8.25 (6.78-10.20) | 19.95 | 0 |
| DeepLabv3+ | HD95 | 105 | 34.21 (22.83-58.73) | 52.75 (42.85-63.99) | 156.14 | 3 |
| DeepLabv3+ | ASSD | 105 | 10.9 (7.28-19.05) | 17.81 (14.21-22.11) | 57.92 | 3 |
| U-Net | HD95 | 107 | 75.2 (32.54-130.74) | 91.87 (79.41-105.64) | 216.2 | 1 |
| U-Net | ASSD | 107 | 23.88 (12.48-36.22) | 28.93 (24.47-34.22) | 65.25 | 1 |

## External Test 2 (Center 3)

| Model | Metric | n | Median (IQR) px | Mean (95% CI) px | p95 px | Undefined |
|---|---|---|---|---|---|---|
| nnU-Net | HD95 | 94 | 25.89 (16.54-42.42) | 35.0 (29.87-40.47) | 94.93 | 0 |
| nnU-Net | ASSD | 94 | 7.72 (5.48-11.24) | 10.2 (8.85-11.68) | 24.79 | 0 |
| DeepLabv3+ | HD95 | 92 | 55.72 (31.75-89.91) | 81.92 (65.50-100.26) | 250.14 | 2 |
| DeepLabv3+ | ASSD | 92 | 19.25 (10.32-31.87) | 34.51 (23.72-48.08) | 115.11 | 2 |
| U-Net | HD95 | 94 | 99.08 (59.27-143.80) | 109.24 (97.13-123.45) | 225.77 | 0 |
| U-Net | ASSD | 94 | 28.2 (18.16-43.14) | 33.99 (29.51-39.67) | 75.02 | 0 |

## Training (model fit)

| Model | Metric | n | Median (IQR) px | Mean (95% CI) px | p95 px | Undefined |
|---|---|---|---|---|---|---|
| nnU-Net | HD95 | 600 | 4.0 (3.16-5.00) | 5.76 (4.65-7.42) | 9.15 | 0 |
| nnU-Net | ASSD | 600 | 1.46 (1.19-1.84) | 1.73 (1.62-1.87) | 3.03 | 0 |
| DeepLabv3+ | HD95 | 573 | 49.4 (30.02-89.01) | 70.3 (65.23-75.54) | 183.59 | 27 |
| DeepLabv3+ | ASSD | 573 | 17.08 (10.02-28.51) | 24.46 (22.30-26.85) | 69.64 | 27 |
| U-Net | HD95 | 597 | 83.23 (46.56-137.28) | 100.63 (95.38-106.29) | 244.02 | 3 |
| U-Net | ASSD | 597 | 25.83 (15.36-42.85) | 32.88 (30.86-35.09) | 80.32 | 3 |

## Paired comparisons, nnU-Net versus each baseline

Wilcoxon signed-rank on identical cases. Negative difference favours nnU-Net (smaller distance is better).

| Cohort | Metric | Comparison | n paired | nnU-Net | Competitor | Difference | p |
|---|---|---|---|---|---|---|---|
| train | HD95 | nnU-Net vs DeepLabv3+ | 573 | 5.85 | 70.30 | -64.45 | 2.41e-95 |
| train | ASSD | nnU-Net vs DeepLabv3+ | 573 | 1.75 | 24.46 | -22.71 | 1.53e-95 |
| train | HD95 | nnU-Net vs U-Net | 597 | 5.75 | 100.63 | -94.88 | 8.46e-99 |
| train | ASSD | nnU-Net vs U-Net | 597 | 1.73 | 32.88 | -31.15 | 1.86e-99 |
| internal_val | HD95 | nnU-Net vs DeepLabv3+ | 250 | 35.72 | 75.23 | -39.50 | 4.38e-26 |
| internal_val | ASSD | nnU-Net vs DeepLabv3+ | 250 | 9.82 | 24.44 | -14.62 | 1.75e-35 |
| internal_val | HD95 | nnU-Net vs U-Net | 257 | 35.65 | 100.93 | -65.28 | 1.55e-36 |
| internal_val | ASSD | nnU-Net vs U-Net | 257 | 9.81 | 30.85 | -21.04 | 1.98e-40 |
| external_test1 | HD95 | nnU-Net vs DeepLabv3+ | 105 | 30.46 | 52.75 | -22.28 | 5.56e-11 |
| external_test1 | ASSD | nnU-Net vs DeepLabv3+ | 105 | 8.18 | 17.81 | -9.63 | 8.48e-15 |
| external_test1 | HD95 | nnU-Net vs U-Net | 107 | 30.58 | 91.87 | -61.29 | 1.86e-15 |
| external_test1 | ASSD | nnU-Net vs U-Net | 107 | 8.31 | 28.93 | -20.62 | 3.72e-18 |
| external_test2 | HD95 | nnU-Net vs DeepLabv3+ | 92 | 35.17 | 81.92 | -46.75 | 1.66e-10 |
| external_test2 | ASSD | nnU-Net vs DeepLabv3+ | 92 | 10.23 | 34.51 | -24.28 | 1.79e-13 |
| external_test2 | HD95 | nnU-Net vs U-Net | 94 | 35.00 | 109.24 | -74.24 | 5.48e-15 |
| external_test2 | ASSD | nnU-Net vs U-Net | 94 | 10.20 | 33.99 | -23.79 | 5.25e-16 |
