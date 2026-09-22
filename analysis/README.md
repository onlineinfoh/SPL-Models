# Analysis outputs

This directory contains the deposited original statistics. Their code, inputs,
configuration and limitations are mapped in [docs/REPRODUCE.md](../docs/REPRODUCE.md).
The previous README is preserved in [docs/history](../docs/history/2026-09-22-before-scope-correction/analysis/README.md).

~~All outputs are from one locked pipeline~~ — the original outputs here and the
later outputs in `protocol/results/` are different experiments. Do not mix them.

The default classification source remains `binary_classification/predictions_tight/`
(the historical 224 px predictions). Changing inference to 300 px does not
regenerate these statistics. `SPL_PRED_DIR`, `SPL_ARCH`, `SPL_RESULTS` and
`SPL_FIGURES` select alternate inputs/outputs in the shared analysis helpers.
`task1_confidence_intervals.py` and `subgroup_and_precision.py` still carry the
historical operating thresholds; do not use them as final 300 px tables without
updating the operating-point configuration after new tuning-cohort inference.

Segmentation Table 2 uses foreground-only metrics on saved masks at native
resolution. Its `internal_val` row means the separate 257-case tuning cohort;
it is not the training-time metric from nnU-Net `fold_all`.

No analysis or training was rerun in the current update.
