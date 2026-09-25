# The Table 2 false-positive-rate artefact: diagnosis and proof

## What reviewers observed

Reviewer 3 and the Editor (comment 14) noted that the DeepLabv3 false-positive-rate
column cannot be reconciled with the precision and recall reported on the same images.
Reviewer 3 computed that each row implies a lesion-to-image area ratio of roughly 5%
for nnU-Net and U-Net but 69-72% for DeepLabv3, and that DeepLabv3's precision of
0.934 exceeded nnU-Net's 0.925 while its FPR was about 34 times higher.

## Root cause

The three segmentation benchmarks did not use the same metric definition.

- `nnunet_benchmarking.py` and `unet_benchmark.py` call `_binary_batch_metrics`,
  which computes FOREGROUND-ONLY metrics: `fpr = fp / (fp + tn)` on the lesion class.
- `deeplab_benchmark.py` calls `_binary_confusion_metrics`, which computes per-class
  values over a 2x2 confusion matrix and then returns `np.nanmean(...)` over BOTH
  classes, that is, CLASS-MACRO-AVERAGED metrics over {background, lesion}.

For any two-class confusion matrix the macro-averaged FPR is identically equal to
one minus the macro-averaged recall:

    macro_FPR = (FPR_bg + FPR_lesion)/2 = ((1 - recall_lesion) + (1 - recall_bg))/2
              = 1 - macro_recall

So the published DeepLabv3+ 'FPR' column was never a false-positive rate.
It carried no information beyond the recall column already next to it.
This is verifiable directly in the submitted numbers: in every DeepLabv3+ row,
Recall + FPR = 1.0000 exactly.

| Cohort | published Recall | published FPR | sum |
|---|---|---|---|
| Training (model fit) | 0.8451 | 0.1549 | 1.0000 |
| Development/tuning (Center 1) | 0.8633 | 0.1367 | 1.0000 |
| External Test 1 (Center 2) | 0.8713 | 0.1287 | 1.0000 |
| External Test 2 (Center 3) | 0.8454 | 0.1546 | 1.0000 |

## Proof by reproduction

Re-running DeepLabv3+ inference from the locked checkpoint and deliberately applying
the class-macro-averaged definition reproduces the PUBLISHED Table 2 values to three
decimal places, which confirms the diagnosis rather than merely asserting it.
Applying the correct foreground-only definition to the same masks gives very
different values.

| Cohort | published Dice | macro-avg reproduction | corrected foreground Dice | published FPR | macro-avg reproduction | corrected foreground FPR |
|---|---|---|---|---|---|---|
| Training (model fit) | 0.8562 | 0.8562 | **0.7210** | 0.1549 | 0.1548 | **0.00439** |
| Development/tuning (Center 1) | 0.8784 | 0.8783 | **0.7659** | 0.1367 | 0.1367 | **0.00408** |
| External Test 1 (Center 2) | 0.8881 | 0.8881 | **0.7831** | 0.1287 | 0.1287 | **0.00317** |
| External Test 2 (Center 3) | 0.8619 | 0.8619 | **0.7361** | 0.1546 | 0.1546 | **0.00361** |

## Important correction to the Editor's premise: Dice and mIoU were NOT unaffected

Editor comment 14 states that the DeepLabv3 FPR values are hard to reconcile "whereas Dice and mIoU
are internally more coherent", which implies those two columns could be retained as they stand.

That is true for the nnU-Net and U-Net rows. It is **not** true for the DeepLabv3+ row.

All five metrics in every DeepLabv3+ row reproduce under class-macro-averaging, not just FPR.
Dice and mIoU are affected just as much, and in the same direction, because averaging in the
background class (whose Dice and IoU are both close to 1, since background occupies about 95 percent
of the image) inflates them.

| Cohort | Published Dice | Corrected Dice | Published mIoU | Corrected IoU |
|---|---|---|---|---|
| Training | 0.8562 | 0.7210 | 0.8010 | 0.6188 |
| Development/tuning | 0.8784 | 0.7659 | 0.8231 | 0.6642 |
| External Test 1 | 0.8881 | 0.7831 | 0.8397 | 0.6927 |
| External Test 2 | 0.8619 | 0.7361 | 0.8076 | 0.6392 |

Table 2 therefore cannot be repaired by fixing the FPR column alone. The entire DeepLabv3+ row must
be replaced.

## A second, separate problem with the "mIoU" column header

The column labelled "mIoU" does not contain the same quantity in every row.

- nnU-Net and U-Net reported **foreground-only IoU**, that is `TP / (TP + FP + FN)` on the lesion
  class. This is a single-class IoU, not a mean over classes, so the "m" in the header is wrong.
- DeepLabv3+ reported a genuine **mean IoU over both classes**.

Verified against the submitted values on the development/tuning cohort: nnU-Net's published 0.8583
equals the measured foreground IoU of 0.8583, whereas its true macro IoU is 0.9252. DeepLabv3+'s
published 0.8231 equals its measured macro IoU of 0.8230, whereas its foreground IoU is 0.6642.

So one header covered two different definitions, which is the same defect as the FPR column and has
the same fix. The column should be renamed **"IoU (lesion)"** and populated with foreground-only IoU
for all three models, as it now is in `analysis/results/table2_corrected.md`.

## Consequences for the manuscript

1. DeepLabv3+ true lesion Dice on the tuning cohort is 0.7659, not 0.8784.
   The published figure was inflated by averaging in the background class, whose Dice
   is close to 1 because background dominates the image.
2. DeepLabv3+ true lesion FPR is 0.00408, essentially identical to
   nnU-Net's 0.00410. It is NOT 34 times higher. The apparent gap was
   entirely an artefact of the differing definitions.
3. The correction STRENGTHENS the manuscript's conclusion. Under one consistent
   definition nnU-Net leads DeepLabv3+ on tuning-set Dice by 0.154 rather than the 0.042 implied by the
   original table, and leads on every cohort and every overlap metric.
4. The nnU-Net and U-Net rows were already computed correctly and are unchanged;
   only the DeepLabv3+ row was affected.
5. The model should be named DeepLabv3+ throughout. The implemented checkpoint is
   `deeplabv3plus_mobilenet` with output stride 16, not DeepLabv3.

## Reviewer 3's implied area ratio, checked directly

Reviewer 3 inferred the lesion-to-image area ratio implied by each row. The true
measured mean lesion area fraction from the ground-truth masks is:

| Cohort | measured mean lesion area fraction |
|---|---|
| Training (model fit) | 0.0451 |
| Development/tuning (Center 1) | 0.0498 |
| External Test 1 (Center 2) | 0.0468 |
| External Test 2 (Center 3) | 0.0634 |

This is consistent with the roughly 5% that Reviewer 3 derived from the nnU-Net and
U-Net rows, and confirms that the 69-72% implied by the DeepLabv3 row was impossible.
