# Per-model benchmark outputs

Raw output of the three per-model evaluation scripts in `seg-model-training/benchmarking/`.

These files are retained as the original output of those scripts and are **not** the reported
segmentation results. The three scripts do not share a metric definition:

- `metrics_nnunet.txt` and `metrics.txt` (U-Net) report foreground-only (lesion class) metrics.
- `metrics_deeplab.txt` reports metrics macro-averaged over both classes, background and lesion.
  For any two-class confusion matrix the macro-averaged false positive rate equals one minus the
  macro-averaged recall, so the FPR column in that file is not a false positive rate, and its Dice
  and IoU are inflated by averaging in the background class.

The reported segmentation results are produced by `analysis/code/seg_metrics_engine.py`, which
applies one foreground-only definition to all three models at native image resolution. See
`analysis/results/table2_corrected.md` and `analysis/results/table2b_boundary_metrics.md`.
